package edu.stanford.nlp.semeval2015.pit;

import org.apache.commons.lang3.StringUtils;

/**
 * Amazon turk votes for this task
 *
 */
class Label {
  public final int positiveVote;
  public final int negativeVote;

  public Label(String labelStr) {
    String[] parts = StringUtils.split(labelStr, "(,) ");
    this.positiveVote = Integer.parseInt(parts[0]);
    this.negativeVote = Integer.parseInt(parts[1]);
  }

  public int totalVotes() {
    return positiveVote + negativeVote;
  }

  public boolean isTrue() {
    return positiveVote > negativeVote;
  }

  public boolean isDebatable() {
    return positiveVote == 2 && negativeVote == 3;
  }

  public String voteStr() {
    return String.format("(%d,%d)", positiveVote, negativeVote);
  }

  public String toString() {
    if (isDebatable())
      return voteStr() + " debatable";
    return voteStr() + " " + isTrue();
  }

}
