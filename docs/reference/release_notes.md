# Release notes

One entry per benchmark release, newest first. Smoke releases get no
entry. Each entry has:

- the release id, scope and producing commit;
- the previous release it is compared with, if any;
- **compatibility changes** since that release: changes to formats,
  schemas, output paths or APIs, and what an older checkout or release
  needs to work with them;
- **result-affecting changes** since that release: every change to data,
  splits, preprocessing, feature costs, acquisition semantics, classifiers,
  method configuration or metrics, naming the commits and the affected
  datasets, methods and settings, and whether results of the previous
  release remain comparable with this one. If there are none, the entry
  says "None".

A release that corrects another says what it corrects.
[How to write an entry](../how-to/publish_a_benchmark_release.md#5-write-the-release-notes-entry);
[the two kinds of change](../explanation/benchmark_releases.md#compatibility-and-comparability).

No benchmark release has been published yet.
