## VAP: Voice Activity Projection


* Blue (**A**) and Yellow (**B**) lines indicate the next speaker probability for that speaker
  - During current activity it is used to predict upcoming SHIFT
  - During silence it is used to predict SHIFT/HOLD
* Green
  - Probability of Backchannel/SHORT
  - During activity it is used to infer SHORT
  - During non-activity (from the specific speaker) it is used to infer BC-prediction


<video width="700" height="400" controls>
  <source src="assets/videos_5/4637.mp4" type="video/mp4">
Your browser does not support the video tag.
</video>



## Resources

* [vap_turn_taking](https://github.com/ErikEkstedt/vap_turn_taking)
  - `VAP`/`Events`/`Metrics`
