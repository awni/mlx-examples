# Video Generation with Open-Sora

This is an example of video generation with Open-Sora in MLX.[^1]

### Setup

Install the dependencies:

```bash
pip install -r requirements.txt
```

### Run

```bash
python generate.py \
  --prompt "A beautiful waterfall" \
  --resolution 240p
```

The command takes about a minute on an M2 Ultra and produces a video like the
following:


To see a list of available options run:

```bash
python generate.py -h
```

[^1]: Refer to the [website](https://hpcaitech.github.io/Open-Sora/) and
  [GitHub repo](https://github.com/hpcaitech/Open-Sora) for more details on
  the original model.
