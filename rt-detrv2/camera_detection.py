import cv2
import torch
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor
import numpy as np
from PIL import Image
import time


class RealTimeObjectDetector:
    def __init__(self, model_name="PekingU/rtdetr_v2_r50vd", confidence_threshold=0.5):
        """
        Initialize the real-time object detector

        Args:
            model_name: The model to use for detection
            confidence_threshold: Minimum confidence score for detections
        """
        # Initialize the model and image processor
        self.image_processor = RTDetrImageProcessor.from_pretrained(model_name)
        self.model = RTDetrV2ForObjectDetection.from_pretrained(model_name)

        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # Set confidence threshold
        self.confidence_threshold = confidence_threshold

        # Generate colors for visualization
        self.colors = self._generate_colors(len(self.model.config.id2label))

    def _generate_colors(self, num_classes):
        """Generate distinct colors for different classes"""
        colors = []
        for i in range(num_classes):
            # Generate colors using HSV color space for better distinction
            hue = i / num_classes
            saturation = 0.8
            value = 0.9

            # Convert HSV to RGB
            c = value * saturation
            x = c * (1 - abs((hue * 6) % 2 - 1))
            m = value - c

            if hue < 1 / 6:
                rgb = (c, x, 0)
            elif hue < 2 / 6:
                rgb = (x, c, 0)
            elif hue < 3 / 6:
                rgb = (0, c, x)
            elif hue < 4 / 6:
                rgb = (0, x, c)
            elif hue < 5 / 6:
                rgb = (x, 0, c)
            else:
                rgb = (c, 0, x)

            # Convert to 8-bit RGB
            color = tuple(int((cc + m) * 255) for cc in rgb)
            colors.append(color)

        return colors

    def process_frame(self, frame):
        """
        Process a single frame and return the annotated frame

        Args:
            frame: BGR format frame from OpenCV
        Returns:
            annotated_frame: Frame with detection boxes and labels
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        # Prepare image for the model
        inputs = self.image_processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Process results
        results = self.image_processor.post_process_object_detection(
            outputs,
            threshold=self.confidence_threshold,
            target_sizes=[(pil_image.height, pil_image.width)],
        )[0]

        # Draw detections on the frame
        annotated_frame = frame.copy()
        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            box = [int(i) for i in box.tolist()]
            score = score.item()
            label = label.item()

            # Get color for this class
            color = self.colors[label % len(self.colors)]

            # Draw box
            cv2.rectangle(annotated_frame, (box[0], box[1]), (box[2], box[3]), color, 2)

            # Prepare label text
            label_text = f"{self.model.config.id2label[label]}: {score:.2f}"

            # Calculate text size and position
            (text_width, text_height), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )

            # Draw label background
            cv2.rectangle(
                annotated_frame,
                (box[0], box[1] - text_height - 4),
                (box[0] + text_width, box[1]),
                color,
                -1,
            )

            # Draw label text
            cv2.putText(
                annotated_frame,
                label_text,
                (box[0], box[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

        return annotated_frame

    def start_detection(self, camera_index=0, display_fps=True):
        """
        Start real-time detection from webcam

        Args:
            camera_index: Index of the camera to use
            display_fps: Whether to display FPS counter
        """
        # Initialize video capture
        cap = cv2.VideoCapture(camera_index)

        # Check if camera opened successfully
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return

        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("Starting real-time detection... Press 'q' to quit.")

        # Variables for FPS calculation
        fps = 0
        frame_count = 0
        start_time = time.time()

        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            # Process frame
            annotated_frame = self.process_frame(frame)

            # Calculate and display FPS
            frame_count += 1
            if display_fps:
                if frame_count % 30 == 0:
                    end_time = time.time()
                    fps = frame_count / (end_time - start_time)
                    frame_count = 0
                    start_time = time.time()

                cv2.putText(
                    annotated_frame,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

            # Display the frame
            cv2.imshow("Real-time Object Detection", annotated_frame)

            # Check for 'q' key to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Clean up
        cap.release()
        cv2.destroyAllWindows()


def main():
    # Create detector instance
    detector = RealTimeObjectDetector(
        model_name="PekingU/rtdetr_v2_r50vd", confidence_threshold=0.5
    )

    # Start real-time detection
    detector.start_detection(camera_index=0, display_fps=True)


if __name__ == "__main__":
    main()
