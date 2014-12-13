package edu.stanford.nlp.semeval2015.pit;

import static edu.stanford.nlp.math.ArrayMath.dotProduct;
import static edu.stanford.nlp.math.ArrayMath.norm;
import static edu.stanford.nlp.math.ArrayMath.pairwiseSubtract;
import it.unimi.dsi.fastutil.doubles.DoubleArrayList;

import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.DoubleStream;
import java.util.stream.StreamSupport;

import org.ejml.simple.SimpleMatrix;

import com.google.common.collect.Ordering;
import com.google.common.collect.Sets;
import com.google.common.primitives.Doubles;

import edu.stanford.nlp.ling.RVFDatum;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.OpenAddressCounter;
/**
 * More sophisticated features of the problem
 *
 */
class Features {
  
  static final Function<Iterator<Double>,Double> MAX_POOLER = Ordering.natural()::max;
  static final Function<Iterator<Double>,Double> MIN_POOLER = Ordering.natural()::min;
  static boolean debug = false;
  static final int POOL_SIZE = 8;
  
  static double l2Sim(double[] v1, double[] v2) {
    return norm(pairwiseSubtract(v1,v2));
  }
  

  static double cosineSim(double[] v1, double[] v2) {
    double ret = dotProduct(v1, v2);
    if (ret == 0)
      return 0;
    ret = ret / (norm(v1) * norm(v2));
    // numerical error correction
    if (ret > 1)
      return 1;
    if (ret < -1)
      return -1;
    return ret;
  }
  
  /**
   * 
   * @return raw feature vector for describing the  
   * features whether the two sentences are paraphrases
   */
  static double[] getFeatureVector(ParaphraseProblem p) {

    DoubleArrayList feats = new DoubleArrayList();

    // coarse grained similarity measures
    double[] v1 = p.s1.avgWordVec();
    double[] v2 = p.s2.avgWordVec();
    double l2sim = l2Sim(v1, v2);
    double cosineSim = cosineSim(v1, v2);
    
    // alignment
//    Alignment a = align(p);
//    feats.add(a.maxScore());
    feats.add(l2sim);
    feats.add(cosineSim);
    feats.add(getNERFeatures(p));
    feats.addAll(getBaselineNgramFeatures(p));
//    feats.addAll(getVectorSpaceAlignmentFeatures(p));
    
    return feats.toDoubleArray();
  }
  
  static double getNERFeatures(ParaphraseProblem p) {
    Set<String> ners1 = Sets.newHashSet(p.s1.getNERs());
    Set<String> ners2 = Sets.newHashSet(p.s2.getNERs());
    return Sets.difference(ners1, ners2).size();
  }
  
  static double[][] getPhraseSimMatrix(ParaphraseProblem p,double printHighScorePairsThreshold){
    
    List<Phrase> ph1 = p.s1.getAllPhrases(p.topicName,3);
    List<Phrase> ph2 = p.s2.getAllPhrases(p.topicName,3);
    
    double[][] sim = new double[ph1.size()][ph2.size()];

    for (int i = 0; i < ph1.size(); i++) {
      String w1 = ph1.get(i).surface;
      double[] v1 = ph1.get(i).vectorize();
      
      for (int j = 0; j < ph2.size(); j++) {
        String w2 = ph2.get(j).surface;
        if(w1.equalsIgnoreCase(w2)){
          sim[i][j] = 1; 
        }else{
          double[] v2 = ph2.get(j).vectorize();
          sim[i][j] = cosineSim(v1, v2);
        }
        if (sim[i][j] < 1 && sim[i][j] >= printHighScorePairsThreshold) {
          System.out.println(w1+"~"+w2+":"+sim[i][j]);
        }
      }
    }
    return sim;
  }
  
  /**
   * 
   * @param p
   * @return an asymmetric similarity matrix for exhaustive trigram shingles
   * in both sentences in vector space
   */
  static DoubleArrayList getVectorSpaceAlignmentFeatures(ParaphraseProblem p){

    double printNothingThreshold = 2;
    double[][] sim = getPhraseSimMatrix(p,printNothingThreshold);
    double[][] pooledSim = pool(sim, POOL_SIZE);
    if(debug && !p.label.isTrue()){
      System.out.println(p);
      System.out.println("Matlab matrix:");
      IOs.printMatlabMatrix(pooledSim,System.out);
//      System.out.println(Arrays.deepToString(sim).replaceAll(", \\[", ", \n\\["));
//      System.out.println(Arrays.deepToString(pooledSim).replaceAll(", \\[", ", \n\\["));
      System.out.println();
    }
    DoubleArrayList ret = new DoubleArrayList();
    for(double[] row:pooledSim){
      ret.addAll(Doubles.asList(row));
    }
    return ret;
  }
  
  
  
    
 /* Conform to the classify package feature paradigm
  * features must be invariant w.r.t. sentence pair ordering
  */
 static RVFDatum<Boolean, Integer> getFeatureDatum(ParaphraseProblem p) {
   Counter<Integer> c = new OpenAddressCounter<>();
   double[] v = getFeatureVector(p);
   for (int i = 0; i < v.length; i++) {
     c.setCount(i, v[i]);
   }
   return new RVFDatum<Boolean, Integer>(c, p.label.isTrue());
 }
 
 // Use string features instead of integer features for flexible
 // feature addition, conjunction etc.
 static RVFDatum<Boolean, String> getFeature(ParaphraseProblem p) {
  Counter<String> c = new OpenAddressCounter<>();
  double[] v = getFeatureVector(p);
  for (int i = 0; i < v.length; i++) {
    c.setCount(i+"", v[i]);
  }
  return new RVFDatum<Boolean, String>(c, p.label.isTrue());
}

 /**
  * generates precision/recall and f1 features based on ngrams
  * 
  * @param features
  * @param t1
  * @param t2
  * @param i
  */
 static Set<String> addPRFFeatures(DoubleArrayList features, List<String> t1,
     List<String> t2, int i) {
   Set<String> igrams1 = Sets.newHashSet(Sentence.getNgrams(t1, i + 1));
   Set<String> igrams2 = Sets.newHashSet(Sentence.getNgrams(t2, i + 1));
   Set<String> intersection = Sets.intersection(igrams1, igrams2);
   double intersectSizeInDouble = intersection.size();
   double precision = intersectSizeInDouble / igrams1.size();
   double recall = intersectSizeInDouble / igrams2.size();
   double pr = precision + recall;
   double f1 = pr == 0 ? 0 : 2 * precision * recall / pr;
   features.add(precision);
   features.add(recall);
   features.add(f1);
   // System.out.println("precision"+(i+1)+"gram\t\t" + precision);
   // System.out.println("recall"+(i+1)+"gram\t\t"+ recall);
   // System.out.println("f"+(i+1)+"gram\t\t\t"+ f1);
   return intersection;
 }

 /**
  * Default baseline
  * @param p
  * @return
  */
 static DoubleArrayList getBaselineNgramFeatures(ParaphraseProblem p) {
   return getCoarseAlignmentFeatures(p, 3, s -> s.tokens);
 }

 /**
  * Generates PRF features for up to trigram as well as the stemmed version
  * 
  * @param p
  * @param maxGram
  * @param tokenizer
  * @return
  */
 static DoubleArrayList getCoarseAlignmentFeatures(ParaphraseProblem p, int maxGram,
     Function<Sentence, List<String>> tokenizer) {
   DoubleArrayList features = new DoubleArrayList(maxGram * 3);
   List<String> t1 = tokenizer.apply(p.s1);
   List<String> t2 = tokenizer.apply(p.s2);
   List<String> stemmed1 = p.s1.getStems();
   List<String> stemmed2 = p.s2.getStems();
   for (int i = 0; i < maxGram; i++) {
     addPRFFeatures(features, t1, t2, i);
     addPRFFeatures(features, stemmed1, stemmed2, i);
   }
   return features;
 }
 

 static DoubleStream stream(Iterator<Double> it){
   Iterable<Double> ia = ()->it;
   return StreamSupport.stream(ia.spliterator(), false).mapToDouble(d->d);
 }
 
 static double[][] pool(double[][] raw,int poolSize){
   
   SimpleMatrix m = new SimpleMatrix(raw);
   
   double[][] ret = new double[poolSize][poolSize];

   // Iterates on the pooled matrix
   double rowRatio = (double) m.numRows() / poolSize;
   double colRatio = (double) m.numCols() / poolSize;

   for (int r = 0; r < poolSize; r++) {

     int rowBase = (int) Math.min(m.numRows() - 2, Math.ceil(rowRatio * r));
     int rowMax = (int) Math.min(m.numRows() - 1, Math.ceil(rowRatio * (r + 1)));

     for (int c = 0; c < poolSize; c++) {
       int colBase = (int) Math.min(m.numCols() - 2, Math.ceil(colRatio * c));
       int cMax = (int) Math.min(m.numCols() - 1, Math.ceil(colRatio * (c + 1)));
       // Note the API for indices are inclusive
       ret[r][c] = MAX_POOLER.apply(
           m.iterator(false, rowBase, colBase, rowMax, cMax));
     }
   }
   return ret;
 }
 
  
}
