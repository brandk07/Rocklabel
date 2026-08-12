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

Or use the GUI that will also guide you through the pipeline: `rocklabel dash` → `localhost:8765`

## Notes

- Evaluation is **leave-one-run-out**. Consecutive frames barely move, so a
  random split leaks near-duplicates and inflates scores.
- Deployable models (ONNX + TorchScript + metadata) live in
  `training/exported/`.
- Raw `.mcap` recordings, the point cache and per-run checkpoints are
  gitignored — a clone gets the code and exported models, not the 30 GB of
  sensor data.

## Real-World Use
- Model eval was not just limited to software testing. I built a robot for a testbed to get
  real-world data to help ensure the viability of the models.
  <img width="560" height="995" alt="image" src="https://github.com/user-attachments/assets/e3097039-4095-45a4-ba5a-e7efd5903d1a" />

## Success So far
- During real-world evals with the robot, I have seen a surprising amount of capability given
  the extremely limited data the models have been trained on on so far. In the initial train run, there were
  only 9 x ~45 second clips of manually collected lidar data with a comforter as the
  ground (for simulated uneven terrain) and random everyday objects as obstacles.
- Live Testing with the trained models showed a remarkably accurate segmentation of the
  incoming lidar data.
- Future testing must occur to see how well this may transfer to an environment with lunar
  simulant and rocks,but the signs are promising so far.

 <img width="1769" height="995" alt="image" src="https://github.com/user-attachments/assets/0e885ee3-4977-44fb-993f-4dedea15eda3" />


**[Full documentation → DOCS.md](DOCS.md)**

*Built with AI assistance (Claude Code).*
