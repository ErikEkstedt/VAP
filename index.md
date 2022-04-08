## VAP: Voice Activity Projection


Visualizations for [Voice Activity Projection: Self-supervised Learning of Turn-taking Events]().

**Resources**
* [vap_turn_taking](https://github.com/ErikEkstedt/vap_turn_taking)
  * Contains the **VAP** modules used to create VAP-labels and zero-shot values for turn-taking events
    - Model logits -> turn-taking event probabilities
  * `VAP`, `Events`, `Metrics` 
* [conv_ssl](https://github.com/ErikEkstedt/conv_ssl)
  * Repository used to train the VAP models



### Videos

The model is trained on 10s segments and is unable to processes an entire
dialog in one forward pass. Therefore, each dialog is split into 10s segments,
using a 5s overlap, and processed separately. The output over a complete dialog
is then the concatenation of the last 5s (using the first 5s as context) over all samples.


* Current time step is the <font color="red"> red Line </font>
    * At the current time step the model will have 5-10seconds of context
    * The model will not always have "heard" both speakers during this time
* <font color="blue"> Blue <strong>A</strong> curve </font>
    * indicate the next speaker probability for **A**
    * During current activity it is used to predict upcoming **SHIFT**
    * During silence it is used to predict **SHIFT**/**HOLD**
* <font color="orange"> Yellow <strong>B</strong> curve </font>
    * indicate the next speaker probability for **B**
    * During current activity it is used to predict upcoming **SHIFT**
    * During silence it is used to predict **SHIFT**/**HOLD**
* <font color="green"> Green Lines </font>
    * Probability of **Backchannel**/**SHORT**
    * During activity it is used to infer **SHORT**
    * During non-activity (from the specific speaker) it is used to infer **BC-prediction**
* The **center box** is a visualization of the **VAP-window**
    * The bins correspond to `[0.2, 0.4, 0.6, 0.8]` seconds
    * At each time step the model outputs a probability distribution over all VAP-window states. The transparency level of the bin colors are the weighted contribution over all states.
      * Full colors -> higher probability of activity
      * White/dim colors -> lower probability of activity


### Highlights
TBA

### Full Dialogs


##### 4637
<video width="90%" height="300" controls>
  <source src="assets/videos_5/4637.mp4" type="video/mp4">
Your browser does not support the video tag.
</video>

##### 4638
<video width="90%" height="300" controls>
  <source src="assets/videos_5/4638.mp4" type="video/mp4">
Your browser does not support the video tag.
</video>

##### 4640
<video width="90%" height="300" controls>
  <source src="assets/videos_5/4640.mp4" type="video/mp4">
Your browser does not support the video tag.
</video>

##### 4801
<video width="90%" height="300" controls>
  <source src="assets/videos_5/4801.mp4" type="video/mp4">
Your browser does not support the video tag.
</video>
