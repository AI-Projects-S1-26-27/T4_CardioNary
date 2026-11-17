# RSNA age parsing fix

**Date:** 2026-11-17
**Branch:** `lung-vedansh-fix-rsna-age`

The RSNA DICOM headers in this dataset store ages as plain digit strings (for
example, `57`), while `parse_age` only accepted values with a DICOM unit suffix
such as `057Y`. The parser therefore turned valid ages into missing values.

The parser now treats a plain one-to-three digit value as years and keeps its
existing `Y`, `M`, `W`, and `D` conversions. The manifest was rebuilt with the
metadata-only mode, which reads DICOM headers and reuses the existing PNG and
96×96 cache files. The rebuilt manifest contains 14,860 patients and has an age
for every included patient; the builder skipped 3 selected rows under its
existing filtering/error rules.

The change also adds parser and header-reading unit tests. Test execution is
pending on the contributor's machine. The RSNA metadata vote and fusion result
files must be regenerated after the matching model and frozen fusion parameters
are confirmed; this note does not claim new evaluation metrics.
The S4 robustness runner reads Kermany fold 0, not the RSNA manifest, so this
metadata-only change does not alter its inputs or reported values.

**AI assistance:** Codex helped inspect the repository, identify the parser
assumption, and draft the code and tests. The contributor should review the diff
and run the commands provided with the pull request before merging.
