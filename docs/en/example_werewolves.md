# Werewolves




## result

Qwen2-7B is able to reach 60% percent win rate in about 20 steps.
<div align="center">
<img width="500" alt="image" src="https://github.com/user-attachments/assets/82019d9d-baed-4868-91cf-29f3a404e787"/>
    <p style="margin-top: 10px; color: #666; font-size: 14px;">
    <em>Reference training process</em>
  </p>
</div>

### Behavior Shifts


Significant role-playing improvement is observed during the experiment.
1. For example, when voted out, the original model trends to reveal its identity as `werewolf`, but fine-tuning, agent will try to cheat its opponents and protect teammates. For example:


<div align="center">
<img width="500" alt="image" src="https://github.com/user-attachments/assets/2c082eb7-8925-47c0-9bf4-e43e9d0732c2"/>
    <p style="margin-top: 10px; color: #666; font-size: 14px;">
    <em>Speech when voted out</em>
  </p>
</div>

2. Agent develop multiple strategy for winning.
<div align="center">
<img width="500" alt="image" src="https://github.com/user-attachments/assets/15e5df78-94ea-4872-90d4-9a21529c85fa"/>
    <p style="margin-top: 10px; color: #666; font-size: 14px;">
    <em>Agents develop multiple strategy such as misleading other players</em>
  </p>
</div>

3. Sometime agents can take advantage of suspect between non-werewolf players to eliminate opponents.

<div align="center">
<img width="500" alt="image" src="https://github.com/user-attachments/assets/388829d1-ae24-458a-971a-3b96fd526063"/>
    <p style="margin-top: 10px; color: #666; font-size: 14px;">
    <em>Taking advantage of suspect</em>
  </p>
</div>

### Expanding Qwen2-7B to Qwen2-14B


<div align="center">
<img width="500" alt="image" src="https://github.com/user-attachments/assets/02739f10-7bba-4bc9-a689-711efb3ba91b"/>
    <p style="margin-top: 10px; color: #666; font-size: 14px;">
    <em>The training process of Qwen2-14B is more robust than Qwen2-7B</em>
  </p>
</div>


