In this tutorial, we introduce the way to debug the workflow and the training algorithms.

## Workflow Debugging

1. Install VSCode and connect to GPU server.

VSCode is a open-source software and provides all debugging plugins for free. Therefore, we choose VSCode as our debugging platform.

VSCode can connect to remote ssh server and operate it as if it is your local machine.
For more details, please refer to VSCode official documents.

2. Install VSCode Python Extension Bundle


3. Create `.vscode/launch.json`. If `.vscode` does not exists yet, create it.


4. Copy and paste the following configuration into `launch.json`

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python Debugger: Launch rollout",
      "type": "debugpy",
      "request": "launch",
      "module": "ajet.cli.launcher",
      "console": "integratedTerminal",
      "args": [
        "--backbone", "debug",
        "--conf", "./path/to/yaml.yaml"
      ],
      "env": {}
    }
  ]
}
```

5. Modify `./path/to/yaml.yaml` field to your task yaml.


6. For more sophisticated task with additional external service, add env variables or more args. For example, if your original training command is:

```bash
export DASHSCOPE_API_KEY="sk-abcdefg"
ajet --conf tutorial/example_appworld/appworld.yaml --with-appworld --backbone='verl'
```

Then, the modified launch.json will be

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python Debugger: Launch rollout",
      "type": "debugpy",
      "request": "launch",
      "module": "ajet.cli.launcher",
      "console": "integratedTerminal",
      "args": [
        "--backbone", "debug",  // verl -> debug
        "--conf", "tutorial/example_appworld/appworld.yaml",
        "--with-appworld",
      ],
      "env": {
        "DASHSCOPE_API_KEY": "sk-abcdefg"
      }
    }
  ]
}
```

7. Press `F5` to start debugging.

8. You can set breakpoint inside the workflow to observe program execution now.