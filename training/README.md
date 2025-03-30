<div id="toc">
  <ul style="list-style: none">
    <summary>
      <h1>
        CollisionSense
      </h1>
    </summary>
  </ul>
</div>

---

**Collison Sense** is a early-collision warning system built on python and rust. It is designed to be used in cars using the raspberry pi (not supported yet).

<div id="toc">
  <ul style="list-style: none">
    <summary>
      <h2>
        Training Process
      </h2>
    </summary>
  </ul>
</div>

**CollisionSense** is developed on YOLOv11n and uses clever math to gague distances and relative position

### 1. Dataset

- We use the BDK100K to train our models (might change in the future)

### 2. Data Preprocessing

- Clean up data
- Make it YOLO compatible

### 4. Training

- Trained on my RTX 4060
- Optimized for ARM architecture

---

> **Note:** The training pipeline is still under active development.
