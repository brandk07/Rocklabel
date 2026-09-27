# Rocklabel

Deep-learning LiDAR rock perception for the **NASA Lunabotics** competition.

This project contains the full pipeline for collecting data with built in SLAM, labeling the collected point clouds in Open3D, generating datasets from the labelled data for training, training different models such as PointNet and PointNet++ as both sliding window classifiers and segmentation based, comparing trained models, and running the models live on incoming scan data. 

Reads ROS 2 `rosbag2` mcaps and native `lidarrig` recordings automatically—no ROS 2 install required.

---

## Installation

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dash,train]'
```

## Quick Pipeline

The core workflow goes from raw capture to live inference in seven steps:

```bash
rocklabel record recordings/volleyball/raw/RUN.mcap --source udp   # 1. capture
rocklabel slam recordings/volleyball/raw/RUN.mcap                  # 2. solve poses
rocklabel label recordings/volleyball/reslam/RUN.reslam.mcap       # 3. click rocks
rocklabel generate recordings/volleyball/reslam/RUN.reslam.mcap \
    --profile full-sweep                                           # 4. build dataset
rocklabel coverage datasets/full-sweep/volleyball                  # 4b. is every rock in it?
rocklabel-train cache                                              # 5. pool it
rocklabel-train compare                                            # 6. train + evaluate
rocklabel live --source udp --model best.pt                        # 7. live inference
```

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

<img width="442" height="248" alt="Labeling GUI - Height Mapping" src="https://github.com/user-attachments/assets/1360e401-3b8d-4711-9a56-4f3f02132882" />
<img width="442" height="248" alt="Labeling GUI - Relief Mapping" src="https://github.com/user-attachments/assets/9c7e4200-5ed2-48d6-bb65-4b58cd2fde4d" />

Evaluation is strictly **leave-one-run-out**. Consecutive frames barely move, so a random split would leak near-duplicates and inflate scores. Deployable models are saved to `training/exported/`.

---

## Real-World Use & Data Collection

Model evaluation was not limited to software testing with fabricated data. To simulate uneven lunar terrain, data was collected in various environments, including a sand volleyball court scattered with obstacle rocks.

<img width="442" height="248" alt="Volleyball Court Environment" src="https://github.com/user-attachments/assets/1afc3d59-2e2d-439b-a0fd-fc814e3003f7" />

---

## Success So Far

During real-world evaluations with the robot, the models have shown remarkably accurate segmentation of the incoming LiDAR data. This capability is highly surprising given the extremely limited data the models have been trained on so far—the initial training run used only 12 short ~45-second clips of manually collected data featuring ~10 limestone rocks scattered throughout a sand court.

**Live Inference Results:**

One of the early model's being used on the left out run of its training batch:

<img width="442" height="248" alt="Live Replay Reflectivity Map" src="https://github.com/user-attachments/assets/32c68995-bc6e-4a9f-84b2-7c5b08155898" />
<img width="442" height="248" alt="Live Replay Binary Segmentation" src="https://github.com/user-attachments/assets/94ed6baf-2d46-4f1d-adda-6273b70748cb" />

Running with live lidar on the robot. This was from a training run on the volleyball court test set, so it transfers well to different a flat terrain environment:

<img width="442" height="248" alt="image" src="https://github.com/user-attachments/assets/b9fd2db8-34b9-4bc7-afdd-03e0b68273de" />
<img width="480" height="270" alt="Screenshot from 2026-09-26 13-58-25" src="https://github.com/user-attachments/assets/8e613ac6-aa57-4a79-bc28-bad9289031a3" />


Future testing must occur to see how well this transfers to a competition environment with actual lunar stimulant, but early signs are very promising.

---

## Hardware Testbed Build

To validate the models outside of pure software evaluation, I designed and built a custom two-wheeled autonomous rover testbed from scratch. The chassis is constructed from slotted flat angle steel for a rigid frame. Electrically, it runs on a 3S LiPo power system and uses an ESP32 microcontroller paired with CAN bus transceivers and motor controllers to drive DC gear motors with encoders.

This setup allows me to replicate closed-loop control and gather realistic, live LiDAR data on the fly to help ensure the viability of the models in the real world.

<img width="248" height="442" alt="Robot Testbed" src="https://github.com/user-attachments/assets/91364a2a-3d37-4625-8a48-6180bfa6bc78" />

---

*Built with AI assistance (Claude Code).*
