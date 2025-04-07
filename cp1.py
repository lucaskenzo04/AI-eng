import cv2
import numpy as np

class DriverAssistant:
    def __init__(self, video_path, cascade_path):
        self.cap = cv2.VideoCapture(video_path)
        self.car_cascade = cv2.CascadeClassifier(cascade_path)

    def detectar_faixas(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        height, width = frame.shape[:2]
        mask = np.zeros_like(edges)
        polygon = np.array([[
            (int(0.1 * width), height),
            (int(0.4 * width), int(0.6 * height)),
            (int(0.6 * width), int(0.6 * height)),
            (int(0.9 * width), height)
        ]], np.int32)
        cv2.fillPoly(mask, polygon, 255)
        cropped_edges = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(cropped_edges, 1, np.pi / 180, threshold=50, minLineLength=60, maxLineGap=150)

        curva = "Reta"
        intensidade = ""
        avg_angle = 0

        if lines is not None:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                angles.append(angle)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

            avg_angle = np.mean(angles)

            if avg_angle > 10:
                curva = "Curva à esquerda"
            elif avg_angle < -10:
                curva = "Curva à direita"

            if abs(avg_angle) > 25:
                intensidade = "acentuada"
            elif abs(avg_angle) > 10:
                intensidade = "suave"

        if curva != "Reta":
            texto = f"{curva} ({intensidade})"
            cv2.putText(frame, texto, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        return frame, curva, intensidade

    def detectar_veiculos(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        carros = self.car_cascade.detectMultiScale(gray, 1.1, 3)
        for (x, y, w, h) in carros:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, 'Veículo', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        return frame, len(carros)

    def processar_video(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame, curva, intensidade = self.detectar_faixas(frame)
            frame, total_veiculos = self.detectar_veiculos(frame)

            # HUD com dados informativos
            hud_text = f"Curva: {curva} {intensidade} | Veículos: {total_veiculos}"
            cv2.rectangle(frame, (0, 0), (600, 30), (50, 50, 50), -1)
            cv2.putText(frame, hud_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("Assistente ao Motorista", frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


def main():
    video_path = "project_video.mp4"
    cascade_path = "cars.xml.xml"
    assistente = DriverAssistant(video_path, cascade_path)
    assistente.processar_video()


if __name__ == "__main__":
    main()
