# Motion Mentor

Motion Mentor is a VR training prototype built by Team Butterfly Effect at the Best Tech and Reply Collaboration hackathon on 14 November 2026. The team was Vinoth, Nuriddin, and Abdul.

The prototype compares five simulated trainee recordings with a 4.73-second hip-hop motion-capture benchmark. The browser dashboard shows animated skeletons, accuracy and rhythm by task and body region, and rule-based coaching suggestions for teaching, learning, and assignments.

## Run the demo

Open [dashboard.html](dashboard.html) locally to view the included demo. To regenerate it and the CSV exports, install Python 3.11 or newer and run the following from the project directory.

```sh
python -m pip install -r requirements.txt
python main.py
```

On Windows, you can also double-click `run.bat` after installing dependencies. Use `python main.py --no-open` to regenerate the outputs without launching a browser.

## How it works

1. **ReferenceAgent** reads the normalized benchmark CSV and splits the 143 frames into three tasks.
2. **TraineeSimulator** applies seeded timing shifts and smooth positional deviations, then adjusts their magnitude to reproduce the configured demo scores.
3. **EvaluatorAgent** scores the saved recordings. Accuracy averages clipped joint-position similarity across 17 joints, excluding the fixed hip root. Coherence compares joint-speed profiles using Pearson correlation.
4. **CoachAgent** uses scores and timing offsets to generate suggested drills. The pipeline exports four report CSVs and embeds the demo data in the standalone dashboard.

The five simulated accuracy scores are **58.30%, 65.50%, 90.89%, 80.62%, and 76.55%**, for a cohort mean of **74.37%**. The cognitive-control index is a prototype heuristic combining accuracy (50%), coherence (30%), and consistency across tasks (20%). The prototype uses simulated data and has yet to be validated against live VR recordings or real trainees.

## Project files

- `main.py` contains the Python pipeline.
- `dashboard_template.html` and `dashboard.html` contain the dashboard template and generated demo.
- `hip_hop_benchmark_features.csv` contains benchmark features from the Mixamo hip-hop animation used at the event.
- `data/profiles/` contains the simulated recordings. `data/` also contains the cohort summary, task scores, region scores, and coaching suggestions.
