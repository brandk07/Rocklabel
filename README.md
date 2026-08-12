# rocklabel

Deep-learning LiDAR rock perception for the **NASA Lunabotics** competition.

Label rocks **once** per recording on the fused point cloud, auto-generate
training samples from **every** frame, train a PointNet/PointNet++ classifier,
then run it on the live sensor.

Reads ROS 2 rosbag2 mcaps and native lidarrig recordings — auto-detected, no
ROS 2 install required.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dash,train]'
```

## Pipeline

```bash
rocklabel record recordings/RUN.mcap --source udp          # 1. capture
rocklabel label recordings/RUN.mcap                        # 2. click rocks
rocklabel generate recordings/RUN.mcap --out datasets/D    # 3. build dataset
rocklabel-train compare                                    # 4. train + evaluate
rocklabel live --source udp --model best.pt                # 5. live inference
```

Or drive all of it from one page: `rocklabel dash` → `localhost:8765`

## Notes

- Evaluation is **leave-one-run-out**. Consecutive frames barely move, so a
  random split leaks near-duplicates and inflates scores.
- Deployable models (ONNX + TorchScript + metadata) live in
  `training/exported/`.
- Raw `.mcap` recordings, the point cache and per-run checkpoints are
  gitignored — a clone gets the code and exported models, not the 30 GB of
  sensor data.

**[Full documentation → DOCS.md](DOCS.md)**
