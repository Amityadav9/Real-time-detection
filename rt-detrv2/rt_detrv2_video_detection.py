import cv2
import torch
from transformers import DetrForObjectDetection, DetrImageProcessor
import numpy as np
from PIL import Image
import matplotlib.cm as cm


class VideoObjectDetector:
    def __init__(self, model_name="facebook/detr-resnet-50", confidence_threshold=0.5):
        self.image_processor = DetrImageProcessor.from_pretrained(model_name)
        self.model = DetrForObjectDetection.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.confidence_threshold = confidence_threshold
        self.colors = self._generate_colors(len(self.model.config.id2label))

    def _generate_colors(self, num_classes):
        colormap = cm.get_cmap("tab20", num_classes)
        return [
            tuple(int(c * 255) for c in colormap(i)[:3]) for i in range(num_classes)
        ]

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        inputs = self.image_processor(images=pil_image, return_tensors="pt").to(
            self.device
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.image_processor.post_process_object_detection(
            outputs,
            threshold=self.confidence_threshold,
            target_sizes=[(pil_image.height, pil_image.width)],
        )[0]

        return self._draw_detections(frame, results)

    def _draw_detections(self, frame, results):
        annotated_frame = frame.copy()
        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            box = [int(i) for i in box.tolist()]
            color = self.colors[label % len(self.colors)]
            cv2.rectangle(annotated_frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            label_text = (
                f"{self.model.config.id2label[label.item()]}: {score.item():.2f}"
            )
            cv2.putText(
                annotated_frame,
                label_text,
                (box[0], box[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
        return annotated_frame

    def process_video(self, video_path, output_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (frame_width, frame_height),
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated_frame = self.process_frame(frame)
            out.write(annotated_frame)
            cv2.imshow("Video Object Detection", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()


def main():
    detector = VideoObjectDetector()
    detector.process_video("Studio_Test_01.mp4", "output_video.mp4")


if __name__ == "__main__":
    main()
