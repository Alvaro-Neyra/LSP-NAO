# -*- coding: utf-8 -*-
from naoqi import ALProxy
import vision_definitions
import time
import numpy as np

# Intenta importar OpenCV
try:
    import cv2
except ImportError:
    cv2 = None
    print("WARNING: OpenCV (cv2) no está instalado. No se podrá abrir la ventana de video.")

IP   = "10.190.103.36"   # <-- IP del NAO
PORT = 9559

CAMERA_INDEX   = 0  # 0 = cámara superior, 1 = cámara inferior
RESOLUTION     = vision_definitions.kVGA       # kQQVGA(160x120), kQVGA(320x240), kVGA(640x480)
COLOR_SPACE    = vision_definitions.kBGRColorSpace
FPS            = 15

RUN_SECONDS    = 10
ENABLE_TRACKER = True    # Pon en False si solo quieres ver la cámara

def main():
    # Proxies principales
    vdev   = ALProxy("ALVideoDevice", IP, PORT)
    tts    = ALProxy("ALTextToSpeech", IP, PORT)
    motion = ALProxy("ALMotion", IP, PORT)

    # (Opcional pero recomendable) Gestionar Autonomous Life
    prev_life = None
    life = None
    try:
        life = ALProxy("ALAutonomousLife", IP, PORT)
        prev_life = life.getState()
        life.setState("disabled")
    except:
        pass

    # Despierta motores por si activas tracker (para mover cabeza)
    try:
        motion.wakeUp()
    except:
        pass

    # (Opcional) Preparar tracker de rostro
    track = None
    if ENABLE_TRACKER:
        try:
            track = ALProxy("ALTracker", IP, PORT)
            try:
                track.setMode("Head")  # mover solo cabeza
            except:
                pass
            try:
                track.stopTracker()
            except:
                pass
            try:
                track.unregisterAllTargets()
            except:
                pass
            track.registerTarget("Face", 0.1)  # 10 cm ancho cara aprox.
            track.track("Face")
        except Exception as e:
            print("No se pudo iniciar ALTracker: %s" % e)
            track = None

    # Suscripción a la cámara
    client_name = "pyViewer_%d" % int(time.time())
    handle = vdev.subscribeCamera(client_name, CAMERA_INDEX, RESOLUTION, COLOR_SPACE, FPS)

    if cv2:
        cv2.namedWindow("NAO Camera", cv2.WINDOW_AUTOSIZE)

    tts.say("Mostrando camara por diez segundos.")
    start = time.time()

    try:
        while time.time() - start < RUN_SECONDS:
            # Obtener imagen
            img = vdev.getImageRemote(handle)
            if img is None:
                time.sleep(0.01)
                continue

            # Estructura típica: [w, h, layers, ... , data]
            w = img[0]
            h = img[1]
            data = img[6]  # string de bytes

            # Convertir a numpy BGR
            frame = np.fromstring(data, dtype=np.uint8)
            if frame.size != w*h*3:
                # Inconsistencia, saltar
                continue
            frame = frame.reshape((h, w, 3))

            # (Opcional) Sobreponer info del tracker
            if ENABLE_TRACKER and track is not None:
                try:
                    pos = track.getTargetPosition(0)  # relativo al torso [x,y,z, wx,wy,wz]
                    if isinstance(pos, list) and len(pos) == 6:
                        txt = "Face: x=%.2f y=%.2f z=%.2f m" % (pos[0], pos[1], pos[2])
                    else:
                        txt = "Face: no data"
                except:
                    txt = "Face: no data"

                if cv2:
                    cv2.putText(frame, txt, (10, 25),
                                0, 0.7, (0, 255, 0), 2)

                # Dibujar mira al centro
                cx, cy = w//2, h//2
                if cv2:
                    cv2.drawMarker(frame, (cx, cy), (255, 255, 255),
                                   markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)

            # Mostrar
            if cv2:
                cv2.imshow("NAO Camera", frame)
                # 1ms de espera para refrescar UI; salir si se presiona 'q'
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

        tts.say("Listo.")
    except KeyboardInterrupt:
        print("Interrumpido por el usuario.")
    finally:
        # Limpieza: desuscribir cámara
        try:
            vdev.unsubscribe(handle)
        except:
            pass

        # Parar tracker y limpiar
        if track is not None:
            try: track.stopTracker()
            except: pass
            try: track.unregisterAllTargets()
            except: pass

        # Cerrar motores y UI
        try:
            motion.rest()
        except:
            pass
        if cv2:
            cv2.destroyAllWindows()

        # Restaurar Autonomous Life si lo modificamos
        if life is not None and prev_life is not None:
            try:
                life.setState(prev_life)
            except:
                pass

if __name__ == "__main__":
    main()
