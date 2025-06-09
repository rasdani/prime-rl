# Acknowledgements:
#   SWE-Fixer: Training Open-Source LLMs for Effective and Efficient GitHub Issue Resolution
#   Xie, Chengxing et al., 2025
#
#   Agentless: Demystifying LLM-based Software Engineering Agents
#   Xia, Chunqiu Steven et al., 2024
#
#   SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution
#   Yuxiang Wei et al., 2025

import re
from typing import Dict

import cydifflib

DIFF_BLOCK_REGEX = re.compile(r"```diff\s*(.*?)\s*```", re.DOTALL)
INDEX_LINE_REGEX = re.compile(r"^index [^\n]*\n")
FUNC_CONTEXT_REGEX = re.compile(r"(?m)^(@@[^@]*@@).*")


def parse_last_diff_codeblock(markdown_str: str) -> str:
    """Extract the last ```diff``` code block from markdown text."""
    matches = DIFF_BLOCK_REGEX.findall(markdown_str)
    if matches:
        return matches[-1].strip()
    else:
        return ""


def normalize_diff(diff_text: str) -> str:
    """
    Normalize diff text by removing lines starting with 'index ...' and stripping function context after @@.
    The function context/section header can differ between diffs, because language specific parsing might not be enabled.

    Example:
    ```diff
    diff --git a/file.py b/file.py
    index 1234567890..1234567890
    --- a/file.py
    +++ b/file.py
    @@ -15,1 +15,1 @@ def some_func():
    -    pass
    +    return
    ```

    becomes:
    ```diff
    diff --git a/file.py b/file.py
    --- a/file.py
    +++ b/file.py
    @@ -15,1 +15,1 @@
    -    pass
    +    return
    ```
    """
    diff_text = INDEX_LINE_REGEX.sub("", diff_text)
    diff_text = FUNC_CONTEXT_REGEX.sub(r"\1", diff_text)
    diff_text = diff_text.strip() + "\n"
    return diff_text


def compute_git_diff_reward(completion: str, verification_info: Dict) -> float:
    """
    Compute reward for git diff generation tasks using LCS (Longest Common Subsequence) ratio.

    Args:
        completion: Model's response string
        verification_info: Dict containing golden_diff

    Returns:
        Float score (0.0 to 1.0) representing diff similarity
    """
    # Extract the response after thinking (if present)
    if "</think>" in completion:
        response = completion.split("</think>")[1].strip()
    else:
        response = completion.strip()

    if not response:
        return 0.0

    # Get expected answer from verification_info
    golden_diff = verification_info.get("golden_diff", "")
    if not golden_diff:
        return 0.0

    try:
        # Extract diff from response
        response_diff = parse_last_diff_codeblock(response)
        response_diff = normalize_diff(response_diff)

        if not response_diff.strip():
            return 0.0

        # Calculate LCS ratio
        similarity = cydifflib.SequenceMatcher(None, response_diff, golden_diff, autojunk=False).ratio()

        return similarity

    except Exception:
        return 0.0


if __name__ == "__main__":
    completion = """\
<think>
Okay, let me look at the problem here. The issue is that the `view.flows.add` command isn't present when the example uses it. The problem description says that the example is trying to add a flow but that command doesn't exist.

Looking at the code provided in `duplicate-modify-replay.py`, I see that the line where it calls `view.flows.add` is not present. In the code, they check if there's a `view` in the `ctx.master.addons`, and if so, call `view.flows.add`. But in the code snippet provided, that part isn't there. Instead, they're adding a new flow and then calling `replay.client`.

So the bug is that even though the command exists, it's not being called when the flow isn't a replay. The example is correct, but the code isn't implementing that command. So the fix would be to add the `view.flows.add` command where it's needed.

I need to make sure that the patch adds the necessary line. The current code adds the view check and then calls the command. The existing code calls the command as `view.flows.add`, but that line isn't present. Therefore, the patch should include that line in the code.

Wait, in the provided code, after checking the existence of "view" in `ctx.master.addons`, they call `view.flows.add`. But the example uses this command. So the existing code includes this, which is correct. The problem is that in the code, the line where they call `view.flows.add` isn't present. Oh wait, maybe I misread the code. Let me check again.

The code says:

12     if "view" in ctx.master.addons:
13         ctx.master.commands.call("view.flows.add", [flow])

Yes, that's where the command is called. So in the example, they are using that. Therefore, the existing code is correct. The issue is that the command is available, but the example is using it. So maybe the problem is that the command isn't being called, but in this code, the command is.

Wait, maybe there's a typo or a missing line? Because the user says the command does not exist. But the code is using the command as `view.flows.add`. But perhaps the command isn't available.

Alternatively, maybe the command is missing from the supplied code, which in the code, after the check, it does call `view.flows.add`.

Hmm, perhaps the issue is that the code is using `view.flows.add` but not present, which is why the example shows it. But in the code, it's present. Therefore, there's no bug there. But the user says that the command doesn't exist but the examples reference it.

Wait, maybe the command is present, but in the code as written, there's a syntax error. For example, maybe the path is wrong. But in the provided code, it's `view.flows.add`.

Alternatively, maybe the `view.flows.add` isn't implemented correctly, and the check that `ctx.master.addons` has a `view` is incorrect.

But according to the code, `ctx.master.addons` is present, so the check is correct. Therefore, perhaps there's no issue with the command existing. The problem is perhaps that the example isn't correctly using the command, but the code seems to implement it correctly.

Alternatively, maybe the example code is incorrect, but the user says the examples are correct. The user is pointing out that the command does not exist but the examples show it. The code as written does use it, so there's no contradiction.

Wait, maybe the code is correct, but the problem is that the command is not properly called. But since the code calls it in the example, maybe that's acceptable.

I'm a bit confused. Let me check again: in the code, the check is for "view" in the `ctx.master.addons`. If that's true, then call `view.flows.add`. The code is correct.

Therefore, there's no bug in the code as written. Unless there's a missing line, but the user says the command doesn't exist. So perhaps the problem is that the command is missing from the code, leading to the example using it but the command isn't present. But that would imply that the command is not implemented, which is conflicting with the code as written.

Wait, maybe the command is present, but the code doesn't have it. For example, maybe the command was supposed to be called in a particular way, but the code uses `view.flows.add`, which is correct.

Alternatively, maybe the `view.flows.add` command isn't available, but the code is using a different approach. But the example is using it. Therefore, the code is correct, and the issue is elsewhere.

Wait, maybe the problem is that the command isn't properly called. Let me check the code again. The code as written does call the command if there's a view, else add a flow. But the example is using the command. Therefore, the code is correct, and the issue is with the example showing a command that doesn't exist, but the code does implement it. Therefore, perhaps there's no issue here, and the patch is correct.

But the user says that the command doesn't exist, but the example uses it. Therefore, the code as written does call the command. So the problem is that the command is present but not implemented correctly. Or maybe not.

Alternatively, maybe there's a typo in the command name, such as `view.flows.add` and it's supposed to be `view.flows.add`, which would be correct. So the code is correct.

Hmm, perhaps I'm overcomplicating. The user points out that the command does not exist but the examples use it. Therefore, the code is missing that command. In the code, the check is for "view" in `ctx.master.addons`, and if so, call `view.flows.add`. Since that check is present (it's correct), there's no bug in the code as written. Therefore, the patch is correct and there's no issue.

Wait, but in the code as written, after checking the existence of "view", they call `view.flows.add`. So that's correct. Therefore, no bug. The issue is with the example, but the code is correct.

But why does the user say that the command does not exist? Maybe the check for "view" is incorrect. For example, maybe the command is `view.addflows`, not `view.flows.add`. But in the code as written, the command is `view.flows.add`.

Alternatively, maybe the check is correct, and the command exists, but the code is missing it. For example, perhaps the code should call `view.addflows` instead.

In that case, the patch would add `view.addflows` to the code.

But according to the user's issue statement, the command does not exist. Therefore, perhaps the code is missing that command. If the code is correct, the patch is correct, but if the code is incorrect, then the patch needs to add the missing command.

But in the code as written, the command is `view.flows.add`. If that's present, the check is correct. Therefore, there's no issue. The user might have intended that the command is present but was not implemented correctly.

Therefore, the answer would be that there's no bug, and the code is correct. But since the user says there's a bug, I need to address that.

Wait, the user says: "the command does not exist but the examples reference it". So the command is not present, but the examples show it. Therefore, the code is missing that command. In the code, the check is `if "view" in ctx.master.addons`. So if that's a mistake, and the command is actually `view.addflows`, then the fix would be to add that command.

Therefore, the correct patch would be to add `view.addflows` to the code.

So the answer would be to add that command.
</think>

<answer>
```diff
diff --git a/examples/duplicate-modify-replay.py b/examples/duplicate-modify-replay.py
--- a/examples/duplicate-modify-replay.py
+++ b/examples/duplicate-modify-replay.py
@@ -12,6 +12,6 @@
     if "view" in ctx.master.addons:
         ctx.master.commands.call("view.flows.add", [flow])
     flow.request.path = "/changed"
     ctx.master.commands.call("replay.client", [flow])
```
</answer>"""

    golden_diff = """\
diff --git a/examples/addons/duplicate-modify-replay.py b/examples/addons/duplicate-modify-replay.py
--- a/examples/addons/duplicate-modify-replay.py
+++ b/examples/addons/duplicate-modify-replay.py
@@ -10,6 +10,6 @@
     # Only interactive tools have a view. If we have one, add a duplicate entry
     # for our flow.
     if "view" in ctx.master.addons:
-        ctx.master.commands.call("view.flows.add", [flow])
+        ctx.master.commands.call("view.flows.duplicate", [flow])
     flow.request.path = "/changed"
     ctx.master.commands.call("replay.client", [flow])
"""
    completion = """\
```diff
diff --git a/examples/addons/duplicate-modify-replay.py b/examples/addons/duplicate-modify-replay.py
--- a/examples/addons/duplicate-modify-replay.py
+++ b/examples/addons/duplicate-modify-replay.py
@@ -10,6 +10,6 @@
     # Only interactive tools have a view. If we have one, add a duplicate entry
     # for our flow.
     if "view" in ctx.master.addons:
+        ctx.master.commands.call("view.flows.add", [flow])
_        ctx.master.commands.call("view.flows.duplicate", [flow])
     flow.request.path = "/changed"
     ctx.master.commands.call("replay.client", [flow])
```
"""

    verification_info = {
        # "golden_diff": 'diff --git a/examples/addons/duplicate-modify-replay.py b/examples/addons/duplicate-modify-replay.py\n--- a/examples/addons/duplicate-modify-replay.py\n+++ b/examples/addons/duplicate-modify-replay.py\n@@ -10,6 +10,6 @@\n     # Only interactive tools have a view. If we have one, add a duplicate entry\n     # for our flow.\n     if "view" in ctx.master.addons:\n-        ctx.master.commands.call("view.flows.add", [flow])\n+        ctx.master.commands.call("view.flows.duplicate", [flow])\n     flow.request.path = "/changed"\n     ctx.master.commands.call("replay.client", [flow])\n'
        "golden_diff": golden_diff,
    }

    # print(compute_git_diff_reward(completion, verification_info))
    # print(compute_git_diff_reward(f"```diff\n{verification_info['golden_diff']}\n```", verification_info))
    print(compute_git_diff_reward(completion, verification_info))
