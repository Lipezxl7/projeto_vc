import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
nome = "captura"
caminho = "video.mp4" 
captura = cv2.VideoCapture(caminho)
while True:
    
    sucesso, frame = captura.read()
    if not sucesso:
        break

    frame = cv2.flip(frame, 1)
    # colocando class 0 para ser somente pessoas
    resultados = model.predict(frame, classes=[0], verbose=False)
    total_agora = len(resultados[0].boxes)
    # usando line para deixar a linha mais fina para nao ficar estranho na tela
    frame_desenhado = resultados[0].plot(line_width=2, font_size=0.5)
    
    cv2.putText(frame_desenhado, f"{total_agora}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.namedWindow("captura", cv2.WINDOW_NORMAL)

    cv2.imshow(nome, frame_desenhado)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

captura.release()
cv2.destroyAllWindows()