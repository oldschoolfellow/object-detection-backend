import cv2
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaRelay
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
import asyncio

# FastAPI setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

relay = MediaRelay()
pcs = set()

# Load YOLOv8 face detection model
model_path = hf_hub_download(
    repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt"
)
model = YOLO(model_path)

def detects(frame_bgr):
    # Convert OpenCV BGR to PIL RGB
    image_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    results = model(image_pil)
    detections = Detections.from_ultralytics(results[0])

    for box in detections.xyxy:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame_bgr, "Face", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

    return frame_bgr

# WebRTC video transform track
class VideoTransformTrack(VideoStreamTrack):
    def __init__(self, track):
        super().__init__()
        self.track = track
        self.frame_count = 0

    async def recv(self):
        frame = await self.track.recv()
        self.frame_count += 1
        frame_bgr = frame.to_ndarray(format="bgr24")

        # Skip frames to reduce CPU load
        # if self.frame_count % 2 != 0:
        #     return frame

        # Run detection asynchronously to avoid blocking
        loop = asyncio.get_event_loop()
        processed = await loop.run_in_executor(None, detects, frame_bgr)

        new_frame = frame.from_ndarray(processed, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame

# WebRTC offer handler
@app.post("/offer")
async def offer(request: Request):
    offer_data = await request.json()
    offer = RTCSessionDescription(sdp=offer_data["sdp"], type=offer_data["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("track")
    def on_track(track):
        if track.kind == "video":
            pc.addTrack(VideoTransformTrack(relay.subscribe(track)))

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return JSONResponse({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    })
