package edu.stanford.nlp.semeval2015.pit;

import static com.google.common.collect.Lists.transform;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.ListIterator;
import java.util.Properties;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.apache.commons.lang3.StringUtils;
import org.apache.commons.lang3.mutable.MutableInt;

import com.google.common.base.Supplier;
import com.google.common.base.Suppliers;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.Lists;
import com.google.common.collect.Multimap;
import com.google.common.collect.RangeSet;
import com.google.common.collect.TreeRangeSet;

import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.math.ArrayMath;
import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.AnnotationSerializer;
import edu.stanford.nlp.pipeline.ProtobufAnnotationSerializer;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation;
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations;
import edu.stanford.nlp.trees.LabeledScoredTreeNode;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Functions;
import edu.stanford.nlp.util.IntPair;
/**
 * Models the sentence objects give in the data files
 *
 */
public class Sentence {
  
  private static final Properties props = new Properties(){
    private static final long serialVersionUID = 1L;
  {
    put("annotators", "tokenize, ssplit, pos, lemma, ner, parse, depparse, sentiment");
    put("ssplit.isOneSentence", "true");
    put("tokenize.class", "PTBTokenizer");
    put("tokenize.language", "en");
  }};
  
  private static final AnnotationSerializer serializer = new ProtobufAnnotationSerializer();
  
  private static Supplier<StanfordCoreNLP> pipeline = Suppliers
      .memoize(()-> new StanfordCoreNLP(props));
  
  private Supplier<Annotation> annotationCache = Suppliers.memoize(()->{
    try{
      return annotate(false);
    }catch(Exception e){
      e.printStackTrace();
    }
    return null;
  });
  
  public final String surface;
  /**
   * Raw word structures read form input file
   */
  public List<Phrase> words;
  /**
   * Processed tokens from words
   */
  public List<String> tokens;
  
  List<String> lowercaseTokens;


  /**
   * Effectively a cache for averaged vector
   */
  private Supplier<double[]> avgVec = Suppliers.memoize(() -> {
    double[] vecSum = new double[Phrase.vectorDimension()];
    MutableInt vecCount = new MutableInt(0);

    // Prevent loss of precision
      nonNullWordVecs().forEach(v -> {
        ArrayMath.pairwiseAddInPlace(vecSum, v);
        vecCount.increment();
      });
      return ArrayMath.multiply(vecSum, 1 / vecCount.doubleValue());
    });

  public Sentence(String surface){
    this.surface = surface;
  }
  
  public Sentence(String surface, String taggedSentence) {
    // super(surface);
    this.surface = surface;
    
    words = transform(Arrays.asList(taggedSentence.split(" ")),s->new Phrase(s,true));

    // normalize token representation here
//    tokens = transform(words,w -> w.surface);
    
    tokens = Lists.transform(getWords(),CoreLabel::originalText);
    lowercaseTokens = Lists.transform(tokens,String::toLowerCase);
  }
  
  /**
   * 
   * @return the sentence to get Stanford NLP annotations
   * @throws IOException 
   * @throws ClassNotFoundException 
   */
  public Annotation getDocAnnotation(){
    return annotationCache.get();
  }
  
  Annotation annotate(boolean forceReannotate) throws Exception{
    File cacheFile = getAnnotationCacheFile();
    Annotation a = null;
    if (!forceReannotate && cacheFile.exists()) {
      if(Params.debugMode)
        System.out.print("Checking cache integrity...");
      try {
        a = IOs.readGz(serializer::read,cacheFile).first;
      } catch (Exception e) {
        e.printStackTrace();
      }
      if (a != null) {
        // cache integrity check
        if (surface.equals(a.get(TextAnnotation.class))){
          if(Params.debugMode)
            System.out.println("Passed!");
          return a;
        }
      }
      System.err.println("Either cache collision or integrity check failed.");
    }
    if(Params.debugMode)
      System.out.println("Software annotation");
    final Annotation ann = new Annotation(surface);
    pipeline.get().annotate(ann);
    IOs.writeGz(o->serializer.write(ann,o), cacheFile);
    return ann;
  }
  
  public List<CoreLabel> getWords(){
    return getDocAnnotation().get(TokensAnnotation.class);
  }
  
  /**
   * 
   * @return the Stanford sentence annotation maps
   * @throws IOException 
   * @throws ClassNotFoundException 
   */
  public CoreMap getSentenceAnnotation() throws ClassNotFoundException, IOException{
    return getDocAnnotation().get(SentencesAnnotation.class).get(0);
  }
  
  /**
   * |
   * @return
   * @throws IOException 
   * @throws ClassNotFoundException 
   */
  public SemanticGraph getDependencies() throws ClassNotFoundException, IOException{
    CoreMap sentence = getSentenceAnnotation();
    SemanticGraph ret = sentence.get(CollapsedCCProcessedDependenciesAnnotation.class);
    return ret;
  }
  
  /**
   * Gets the NER label for each word
   * @return
   * @throws ClassNotFoundException
   * @throws IOException
   */
  public List<String> getNERs(){
    return Lists.transform(getWords(), w->w.ner());
  }
  
  /**
   * 
   * @return the sentiment tree of the sentence
   * @throws IOException 
   * @throws ClassNotFoundException 
   */
  public Tree getSentiment() throws ClassNotFoundException, IOException{
    return getSentenceAnnotation().get(SentimentCoreAnnotations.AnnotatedTree.class);
  }
  
  /**
   * 
   * @return root sentiment on scale 0-4
   */
  public int getOverallSentiment(){
    try {
      CoreMap label = (CoreMap) getSentiment().label();
      return label.get(RNNCoreAnnotations.PredictedClass.class);
    } catch (ClassNotFoundException | IOException e) {
      e.printStackTrace();
    }
    return 0;
  }
  
  public List<String> getStems(){
    return Lists.transform(tokens, Phrase::lowercasedStem);
//    Lists.transform(getDocAnnotation().get(TokensAnnotation.class),
//      l->l.get(LemmaAnnotation.class));
  }
  
  private File getAnnotationCacheFile(){
    return new File(Params.cachedAnnotation+IOs.getMD5FileNameFor(surface));
  }
  
  /**
   * @param tokens
   * @param size
   * @return gets the ngrams of a sub-region of the tokens
   */
  public static List<String> getNgrams(List<String> tokens,int size) {
    int numOfGrams = tokens.size() - size + 1;
    if (numOfGrams <= 0)
      return Collections.emptyList();
    List<String> ngrams = Lists.newArrayListWithCapacity(numOfGrams);
    for (int i = 0; i <  numOfGrams; i++) {
      ngrams.add(StringUtils.join(tokens.subList(i, i + size), ' '));
    }
    return ngrams;
  }
  
  /**
   * @param tokens
   * @param from
   * @param to
   * @param size
   * @return gets the ngrams of a sub-region of the tokens
   */
  public static List<String> getNgrams(List<String> tokens, int from, int to,
      int size) {
    int numOfGrams = to - from - size + 1;
    if (numOfGrams <= 0)
      return Collections.emptyList();
    List<String> ngrams = Lists.newArrayListWithCapacity(numOfGrams);
    for (int i = from; i < from + numOfGrams; i++) {
      ngrams.add(StringUtils.join(tokens.subList(i, i + size), ' '));
    }
    return ngrams;
  }

  /**
   * 
   * @param size
   * @return ngrams of the sentence with specified size
   */
  public List<String> getNgrams(int size, Function<String, String> normalizer) {
    List<String> normalized = tokens.stream().map(normalizer)
        .collect(Collectors.toList());
    return getNgrams(normalized, size);
  }

  
  /**
   * @param maxSize
   * @return ngrams up to maxSize
   */
  public List<String> getShingles(int maxSize) {
    return getShingles(maxSize, Functions.identityFunction());
  }
  
  /**
   * 
   * @param maxSize
   * @param normalizer
   * @return ngrams of normalized tokens up to maxSize
   */
  public List<String> getShingles(int maxSize,Function<String,String> normalizer) {
    List<String> shingles = Lists.newArrayList();
    for (int len = 1; len <= maxSize; len++) {
      shingles.addAll(getNgrams(len,normalizer));
    }
    return shingles;
  }

  
  /**
   * Get all shingled phrases up to maxSize length
   * 1. We rewrite entities to the same name in both sentences
   * 2. Only shingles on non-entities are included
   * @param maxSize
   * @return
   */
  public List<Phrase> getAllPhrases(String topic,int maxSize) {
    
    // Extract monolithic phrase first
    List<String> shingles = Lists.newArrayList();
    List<String> topicTokens = Arrays.asList(topic.toLowerCase().split(" "));
    int topicIndex = Collections.indexOfSubList(lowercaseTokens, topicTokens);
    
    
    List<String> nerLabels = Lists.newArrayList(getNERs());
    
    
    if (topicIndex >= 0) {// mark the entity
      for (int i = topicIndex; i < topicIndex + topicTokens.size(); i++) {
        if ("O".equals(nerLabels.get(i)))
          nerLabels.set(i, "TOP");
      }
    }

    // segmentation state variables:
    String prevLabel = nerLabels.get(0);
    List<String> seq = Lists.newArrayList();
    
    for (int i = 0; i < tokens.size(); i++) {
      String curLabel = nerLabels.get(i);
      
      // check if current label is different
      if(!prevLabel.equals(curLabel)){// boundaries
        if(!"O".equals(prevLabel)){// seq is entity
          shingles.add(String.join(" ", seq));
        } else {// seq is non entity
          for (int len = 1; len <= maxSize; len++) {
            shingles.addAll(getNgrams(seq, len));
          }
        }
        seq.clear();
      }
      seq.add(lowercaseTokens.get(i));
    }
    return shingles.stream().map(Phrase::new).collect(Collectors.toList());
  }
  
  /**
   * Simple exhaustive shingle based phrases
   * @param maxSize
   * @return
   */
  public List<Phrase> getAllPhrases(int maxSize) {
    return getShingles(maxSize)
        .stream()
        .map(Phrase::new)
        .collect(Collectors.toList());
  }

  public List<Phrase> getEntitySegmentedPhrases(int maxSize) {
    int start = 0, end = -1;
    List<String> phrases = Lists.newArrayList();
    for (int i = 0; i < words.size(); i++) {
      boolean out = words.get(i).NETag.equals("O");
      if (out) {
        end = i + 1;
      }
      if (!out || i == words.size() - 1) {
        if (end > 0) {
          for (int s = 1; s <= maxSize; s++) {
            phrases.addAll(getNgrams(tokens, start, end, s));
          }
          end = -1;
        }
        start = i + 1;
      }
    }
    phrases.addAll(getEntities().stream().map(s -> s.surface)
        .collect(Collectors.toList()));
    return phrases.stream().map(s -> new Phrase(s, Phrase.vectorize(s)))
        .filter(Phrase::hasKnownWordVector).collect(Collectors.toList());
  }

  /**
   * 
   * @return named entities in sentence
   */
  public List<Span> getEntities() {
    return decode(words, w -> w.NETag).filter(c -> !"O".equals(c.label)).collect(
        Collectors.toList());
  }

  /**
   * 
   * @return shallow parsing chunks
   */
  public List<Span> getChunks() {
    return decode(words, w -> w.chunkTag).collect(Collectors.toList());
  }

  public Stream<double[]> nonNullWordVecs() {
    return words.stream().map(Phrase::vectorize).filter(v -> v != null);
  }

  public double[] avgWordVec() {
    return avgVec.get();
  }

  public String toString() {
    return surface;
  }
  
  public void dispose(){
    annotationCache = null;
  }

  /***
   * BIO decoding utils
   *
   */
  private enum State {
    B, I, O;

    public boolean is(String label) {
      return label.startsWith(name());
    }
  }

  public static class Span {
    public final String label;
    public final String surface;

    public <W> Span(List<W> words, Function<W, String> labelOf) {
      surface = StringUtils.join(words, ' ');
      String tempLabel = labelOf.apply(words.get(0));
      if (tempLabel.length() >= 2)
        label = StringUtils.substringAfter(tempLabel, "-");
      else
        label = tempLabel;
    }

    public String toString() {
      return String.format("[%s] %s", label, surface);
    }
  }

  /**
   * Groups BIO labeled entities into groups
   * 
   * @param rawElements
   * @return
   */
  public static Stream<Span> decode(List<Phrase> rawElements,
      Function<Phrase, String> labelOf) {

    Stream.Builder<Span> grouping = Stream.builder();
    List<Phrase> temp = Lists.newArrayList();
    for (Phrase w : rawElements) {
      String label = labelOf.apply(w);
      // If it is not inside, we are on a segment boundary
      if (!State.I.is(label)) {
        if (temp.size() > 0) {
          grouping.add(new Span(temp, labelOf));
          temp.clear();
        }
      }
      temp.add(w);
    }
    if (!temp.isEmpty())
      grouping.add(new Span(temp, labelOf));
    return grouping.build();
  }

  public int length() {
    return tokens.size();
  }
}
