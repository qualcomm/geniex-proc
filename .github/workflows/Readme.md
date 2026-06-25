# List of workflows and actions
This folder contains workflows that are helpful for maintaining a smooth and secure development process. The workflows should be enabled for open-source projects.

Workflows:
1. `qcom-preflight-checks.yml` - This workflow runs several preflight checks, including copyright, email, repolinter, and security checks.  See [qualcomm/qcom-actions](https://github.com/qualcomm/qcom-actions)
2. `stale-issues.yaml` - This workflow runs daily to check for stalled issues and PRs. Issues/PRs with no activity for 30 days are marked stale with a comment to draw attention.
3. `build-check.yml` - On every PR and push to `main`, compiles `geniex-proc` + `geniex-proc-vision` on the GitHub-hosted `windows-latest` runner.

