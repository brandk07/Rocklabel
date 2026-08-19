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
rocklabel record recordings/volleyball/raw/RUN.mcap --source udp   # 1. capture
rocklabel slam recordings/volleyball/raw/RUN.mcap                  # 2. solve poses
rocklabel label recordings/volleyball/reslam/RUN.reslam.mcap       # 3. click rocks
rocklabel generate recordings/volleyball/reslam/RUN.reslam.mcap \
    --profile full-sweep                                           # 4. build dataset
rocklabel-train cache                                              # 5. pool it
rocklabel-train compare                                            # 6. train + evaluate
rocklabel live --source udp --model best.pt                        # 7. live inference
```

Or use the GUI that will also guide you through the pipeline: `rocklabel dash` → `localhost:8765`

## Layout

Every folder puts the thing that produced a file into the path:

```
recordings/<project>/{raw,reslam}/   raw captures, and re-solved trajectories
labels/<project>/                    hand-placed rock labels
datasets/<profile>/<run>/            training frames, named by HOW they were cut
training/caches/<profile>/           pooled samples, one cache per profile
training/experiments/<exp>/<arm>/    every trained fold
training/reports/<exp>/              figures and tables
training/exported/                   deployable models (ONNX + TorchScript)
```

`<profile>` is a **generation profile** — a named way of cutting a recording
into frames (`full-sweep` is the default and the one to use). See
[DOCS.md](DOCS.md#generation-profiles).

Every folder under `training/` has a README explaining what is in it; start at
[training/README.md](training/README.md).

## Code layout

```
rocklabel/recording/   reading recordings, getting points and poses out
rocklabel/geometry/    headless maths on a point cloud
rocklabel/dataset/     labeled recording -> training data
rocklabel/gui/         every Open3D window
rocklabel/slam/        offline trajectory solver
rocklabel/live/        the live sensor rig
rocklabel/train/       training and evaluation (the only place torch is used)
rocklabel/dashboard/   the web UI
```

Full map, and the rules worth knowing before editing:
[rocklabel/README.md](rocklabel/README.md).

## Notes

- Evaluation is **leave-one-run-out**. Consecutive frames barely move, so a
  random split leaks near-duplicates and inflates scores.
- Deployable models (ONNX + TorchScript + metadata) live in
  `training/exported/`.
- Raw `.mcap` recordings, the point cache and per-fold checkpoints are
  gitignored — a clone gets the code, the dataset manifests, the written-up
  results and the exported models, not the 30 GB of sensor data or the 4 GB of
  checkpoints.

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
