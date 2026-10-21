# lukepopler.me

My personal site. Just HTML and CSS, no framework, hosted on Vercel.

Every page is a folder with its own index.html:

- `/` - home page
- `/running` - heatmap of every run from my COROS watch
- `/runscore` - RunScore paper
- `/recovhr` - RecovHR paper and calculator
- `/multi` and `/IBhistory` - class pages

The heatmap data comes from a Strava export, built with `_tools/heatmap/build_heatmap.py`.

To run it locally:

```
python3 -m http.server
```
