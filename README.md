# Rocklabel

Deep-learning LiDAR rock perception for the **NASA Lunabotics** competition.

Label rocks **once** per recording on the fused point cloud, auto-generate training samples from **every** frame, train a PointNet/PointNet++ classifier, then run it on the live sensor.

Reads ROS 2 `rosbag2` mcaps and native `lidarrig` recordings automatically—no ROS 2 install required.

---

## Installation

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dash,train]'
```

## Quick Pipeline

The core workflow goes from raw capture to live inference in five steps:

```bash
rocklabel record recordings/RUN.mcap --source udp          # 1. capture
rocklabel label recordings/RUN.mcap                        # 2. click rocks
rocklabel generate recordings/RUN.mcap --out datasets/D    # 3. build dataset
rocklabel-train compare                                    # 4. train + evaluate
rocklabel live --source udp --model best.pt                # 5. live inference
```

> **Tip:** You can also drive everything from the web dashboard. Run `rocklabel dash` and open `http://localhost:8765` in your browser.

---

## Tech Stack

```text
Software & Machine Learning
• Languages & Frameworks: Python, PyTorch (PointNet / PointNet++ models)
• 3D Perception & Math: Open3D, NumPy, SciPy
• Data & Middleware: ROS 2 (rosbag2 / MCAP), raw UDP packet parsing
• Tooling: Flask (for the interactive web dashboard)
```

---

## Workflow & Labeling

Labeling is done via an interactive 3D GUI. You can drop bounding shapes (spheres, boxes, lassos) on the fused cloud, set crop limits, and adjust reflectivity ranges to build your dataset.

![Labeling GUI - Height Mapping](https://github.com/user-attachments/assets/1360e401-3b8d-4711-9a56-4f3f02132882)
![Labeling GUI - Relief Mapping](https://github.com/user-attachments/assets/9c7e4200-5ed2-48d6-bb65-4b58cd2fde4d)

Evaluation is strictly **leave-one-run-out**. Consecutive frames barely move, so a random split would leak near-duplicates and inflate scores. Deployable models (ONNX + TorchScript + metadata) are saved to `training/exported/`.

---

## Real-World Use & Data Collection

Model evaluation was not limited to software testing with fabricated data. To simulate uneven lunar terrain, data was collected in various environments, including a sand volleyball court scattered with obstacle rocks.

![Volleyball Court Environment](https://github.com/user-attachments/assets/1afc3d59-2e2d-439b-a0fd-fc814e3003f7)

---

## Hardware Testbed Build

To validate the models outside of pure software evaluation, I designed and built a custom two-wheeled autonomous rover testbed from scratch. The chassis is constructed from slotted flat angle steel for a rigid frame. Electrically, it runs on a 3S LiPo power system and uses an ESP32 microcontroller paired with CAN bus transceivers and motor controllers to drive DC gear motors with encoders.

This setup allows me to replicate closed-loop control and gather realistic, live LiDAR data on the fly to help ensure the viability of the models in the real world.

![Robot Testbed](https://github.com/user-attachments/assets/91364a2a-3d37-4625-8a48-6180bfa6bc78)

---

## Success So Far

During real-world evaluations with the robot, the models have shown remarkably accurate segmentation of the incoming LiDAR data. This capability is highly surprising given the extremely limited data the models have been trained on so far—the initial training run used only 12 short ~45-second clips of manually collected data featuring ~10 limestone rocks scattered throughout a sand court.

**Live Inference Results:**

![Live Replay Reflectivity Map](https://github.com/user-attachments/assets/32c68995-bc6e-4a9f-84b2-7c5b08155898)
![Live Replay Binary Segmentation](https://github.com/user-attachments/assets/94ed6baf-2d46-4f1d-adda-6273b70748cb)

Future testing must occur to see how well this transfers to a competition environment with actual lunar simulant, but early signs are very promising.

---

**[Full documentation → DOCS.md](DOCS.md)**

*Built with AI assistance (Claude Code).*
