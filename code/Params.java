package edu.stanford.nlp.semeval2015.pit;

import java.net.InetAddress;
import java.net.UnknownHostException;

/**
 * Misc. constants and parameters for this project
 *
 */
class Params {

  public static boolean debugMode = false;
  
  public static String homeDir = getHomeDir();
  public static String trainingData = homeDir + "data/train.data";
  public static String testingData = homeDir + "data/dev.data";
  public static String w2vDir = homeDir + "/wordVectors/word2vec/";
  public static String wordVectorFile = homeDir
//      + "wordVectors/word2vec/GoogleNews-vectors-negative300-headless.txt";
  // + "wordVectors/glove.twitter.27B.200d.txt";
  + "wordVectors/word2vec/wiki_giga_vectors.txt";
//  + "wordVectors/word2vec/wiki_giga_vectors.skipGram.txt";
  public static String cachedAnnotation = homeDir + "annotation/";
  public static String socherOutput = homeDir
      + "socher/classifyParaphrases/output.txt";
  public static String depparseModel = "/u/nlp/data/depparser/nn/distrib-2014-10-26/PTB_CoNLL_params.txt.gz";
  
  public static final String latestModels = "/u/nlp/data/StanfordCoreNLPModels/corenlp/stanford-corenlp-2014-10-29-models.jar";
  public static final String gigaWordPath = "/scr/nlp/data/StanfordParsed/gigaword/";
  

  public static String cachedVectorsFile() {
    return wordVectorFile + ".cachedSubset";
  }

  public static String getHostName() {
    try {
      return InetAddress.getLocalHost().getHostName();
    } catch (UnknownHostException e) {
      e.printStackTrace();
    }
    return null;
  }

  public static String getHomeDir() {
    String host = getHostName();
    if ("DEADBEEF".equals(host))
      return "F:/Google Drive/Research/Stanford/semeval2015/";
    if ("Zephex".equals(host))
      return "D:/Google Drive/Research/Stanford/semeval2015/";
    return "/u/melvinj/scr/semeval2015/";
  }

  
}
