# -*- coding: utf-8 -*-
from __future__ import print_function
from naoqi import ALProxy
import vision_definitions
import time
import numpy as np
import cv2
import os

IP   = "10.36.30.90"
PORT = 9559

CFG   = r"D:\yolov3-tiny.cfg"
WEI   = r"D:\yolov3-tiny.weights"
NAMES = r"D:\coco.names"

NMS_THRESH  = 0.4
CONF_THRESH = 0.5

def assert_file_ok(path, min_bytes, allow_html=False):
    if not os.path.isfile(path):
        raise IOError("No se encontró el archivo: %s" % path)
    size = os.path.getsize(path)
    if size < min_bytes:
        raise IOError("Archivo muy pequeño o corrupto: %s (%d bytes)" % (path, size))
    # detectar HTML por descarga equivocada
    with open(path, "rb") as f:
        head = f.read(200)
    hl = head.lower()
    if (b"<html" in hl or b"<!doctype html" in hl) and not allow_html:
        raise IOError("Parece HTML en %s (descarga incorrecta)." % path)

def clean_cfg(src_path):
    """
    Lee el cfg (maneja UTF-16/UTF-8 BOM), normaliza a UTF-8 sin BOM y
    elimina caracteres inválidos. Devuelve la ruta de un archivo temporal limpio.
    """
    with open(src_path, "rb") as f:
        raw = f.read()

    # Detectar UTF-16 LE/BE (común si se guardó como "Unicode" en Notepad)
    # y convertir a utf-8
    def decode_best(b):
        for enc in ("utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "latin-1"):
            try:
                return b.decode(enc)
            except:
                pass
        # último recurso: ignorar errores
        return b.decode("utf-8", "ignore")

    text = decode_best(raw)

    # Mantener solo líneas que Darknet/OpenCV entiende:
    # - vacías
    # - empiezan con '#' (comentario)
    # - secciones [xxx]
    # - key=value
    cleaned_lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            cleaned_lines.append("")  # línea en blanco
            continue
        if s.startswith("#"):
            cleaned_lines.append(s)
            continue
        if s.startswith("[") and s.endswith("]"):
            cleaned_lines.append(s)
            continue
        # parámetros tipo key=value
        if "=" in s:
            # quitar espacios alrededor de '='
            key, val = s.split("=", 1)
            cleaned_lines.append(key.strip() + "=" + val.strip())
            continue
        # cualquier otra cosa la tratamos de comentario para no romper el parser
        cleaned_lines.append("# " + s)

    cleaned_text = "\n".join(cleaned_lines)
    tmp = os.path.join(os.path.dirname(src_path), "_yolov3_tiny_clean.cfg")
    with open(tmp, "wb") as f:
        f.write(cleaned_text.encode("utf-8"))
    return tmp

def load_classes(path):
    with open(path, "r") as f:
        return [l.strip() for l in f.readlines()]

def main():
    # Validaciones de archivos
    assert_file_ok(CFG,   500)       # cfg ~KB
    assert_file_ok(WEI,   1000000)   # pesos ~ decenas de MB
    assert_file_ok(NAMES, 100)       # names > 80 líneas aprox

    # Limpieza/normalización del cfg
    CFG_USE = clean_cfg(CFG)
    print("CFG limpio en:", CFG_USE)
    print("WEI:", WEI)
    print("NAMES:", NAMES)

    # Proxies
    vdev = ALProxy("ALVideoDevice", IP, PORT)
    tts  = ALProxy("ALTextToSpeech", IP, PORT)

    # DNN
    try:
        net = cv2.dnn.readNetFromDarknet(CFG_USE, WEI)
    except Exception as e:
        raise RuntimeError("Fallo al cargar Darknet con cfg/pesos: %s" % e)

    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    ln = net.getLayerNames()
    # getUnconnectedOutLayers devuelve diferentes formas en 3.4.x
    out_idx = net.getUnconnectedOutLayers()
    try:
        out_idx = out_idx.flatten()
    except:
        pass
    out_layers = [ln[i - 1] if isinstance(i, (int, long)) else ln[i[0] - 1] for i in out_idx]

    classes = load_classes(NAMES)

    # Cámara NAO
    handle = vdev.subscribeCamera("yolo_viewer", 0, vision_definitions.kVGA,
                                  vision_definitions.kBGRColorSpace, 10)
    cv2.namedWindow("NAO OBJ", cv2.WINDOW_AUTOSIZE)

    last_said = ""
    cooldown_s = 3.0
    last_time  = 0.0

    tts.say("Iniciando detección de objetos.")
    try:
        while True:
            img = vdev.getImageRemote(handle)
            if img is None:
                continue

            w, h, data = img[0], img[1], img[6]
            frame = np.fromstring(data, dtype=np.uint8)
            if frame.size != w*h*3:
                continue
            frame = frame.reshape((h, w, 3))

            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            outs = net.forward(out_layers)

            boxes, confs, class_ids = [], [], []
            for out in outs:
                for det in out:
                    scores = det[5:]
                    cid = int(np.argmax(scores))
                    conf = float(scores[cid])
                    if conf > CONF_THRESH:
                        cx, cy, bw, bh = det[0]*w, det[1]*h, det[2]*w, det[3]*h
                        x = int(cx - bw/2); y = int(cy - bh/2)
                        boxes.append([x, y, int(bw), int(bh)])
                        confs.append(conf)
                        class_ids.append(cid)

            idxs = cv2.dnn.NMSBoxes(boxes, confs, CONF_THRESH, NMS_THRESH)
            label_to_say = None
            best_conf = 0.0
            if len(idxs) > 0:
                try:
                    flat = idxs.flatten()
                except:
                    flat = idxs
                for i in flat:
                    x,y,bw,bh = boxes[i]
                    cls = classes[class_ids[i]]
                    conf = confs[i]
                    cv2.rectangle(frame, (x,y), (x+bw,y+bh), (0,255,0), 2)
                    cv2.putText(frame, "%s %.2f" % (cls, conf), (x, max(20, y-10)), 0, 0.5, (0,255,0), 2)
                    if conf > best_conf:
                        best_conf = conf
                        label_to_say = cls

            now = time.time()
            if label_to_say and (label_to_say != last_said or now - last_time > cooldown_s):
                # mapeo básico EN->ES (puedes ampliarlo)
                es_map = {
                    "person":"persona","book":"cuaderno","bottle":"botella",
                    "cell phone":"celular","laptop":"laptop","keyboard":"teclado",
                    "mouse":"mouse","tv":"televisor","backpack":"mochila"
                }
                lab_es = es_map.get(label_to_say, label_to_say)
                art = "una" if lab_es.endswith("a") else "un"
                tts.say("Este es %s %s." % (art, lab_es))
                last_said = label_to_say
                last_time = now

            cv2.imshow("NAO OBJ", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27 or k == ord('q'):
                break

        tts.say("Detección finalizada.")
    finally:
        try: vdev.unsubscribe(handle)
        except: pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
