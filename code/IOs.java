package edu.stanford.nlp.semeval2015.pit;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.io.Writer;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.LinkedList;
import java.util.List;
import java.util.Scanner;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

import com.google.common.collect.Lists;
import com.google.common.hash.Hashing;
import com.google.common.primitives.Doubles;

import edu.stanford.nlp.narratives.GigaDocReader;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import edu.stanford.nlp.stats.OpenAddressCounter;
import edu.stanford.nlp.util.StringUtils;
import edu.stanford.nlp.util.Timing;

/**
 * Convenience class for some Java 8 I/O as well as some methods for this
 * specific task
 *
 */
public class IOs {

  public static void writeLines(String location, Stream<String> lines)
      throws IOException {
    writeLines(location, (Iterable<String>) lines::iterator);
  }

  public static void writeLines(String location, Iterable<String> lines)
      throws IOException {
    Files.write(Paths.get(location), lines);
  }

  private static PrintStream nullStream = new PrintStream(new OutputStream() {
    @Override
    public void write(int arg0) throws IOException {

    }
  });

  private static PrintStream syso = System.out;
  private static PrintStream syse = System.err;

  public static void silenceConsole() {
    System.setOut(nullStream);
    System.setErr(nullStream);
  }

  public static void silenceErr() {
    System.setErr(nullStream);
  }

  public static void enableConsole() {
    System.setOut(syso);
    System.setErr(syse);
  }

  public static String getMD5FileNameFor(String content) {
    return Hashing.md5().hashString(content, Charset.defaultCharset()).toString();
  }

  public static void preprocessTwitterData() {
    GigaDocReader reader = new GigaDocReader("");

  }

  public static <T> T readGz(ThrowableFunction<InputStream, T> reader, File cacheFile)
      throws Exception {
    InputStream is = new GZIPInputStream(new FileInputStream(cacheFile));
    T ret = reader.apply(is);
    is.close();
    return ret;
  }

  public static <T> void writeGz(ThrowableConsumer<OutputStream> writer,
      File cacheFile) throws Exception {
    OutputStream os = new GZIPOutputStream(new FileOutputStream(cacheFile));
    writer.accept(os);
    os.close();
  }

  static Counter<String> dumpShingles(String saveToFile) throws IOException {

    Counter<String> shingles = new OpenAddressCounter<String>();

    ParaphraseProblem.readAllProblems().stream()
        .flatMap(ParaphraseProblem::getShingleStream).map(s -> s.replace(' ', '_'))
        .forEach(shingles::incrementCount);
    
    if (saveToFile != null){
      Stream<String> countLines = Counters.toSortedListWithCounts(shingles).stream()
          .map(p -> p.first + " " + p.second.intValue());
      IOs.writeLines(Params.homeDir + saveToFile, countLines);
    }
    return shingles;
  }
  
  
  static void saveProblemsForSocher() throws IOException{
    List<ParaphraseProblem> problems = ParaphraseProblem.readDefinite(Params.testingData);
    try(final PrintWriter w = new PrintWriter("/user/xiao2/scr/semeval2015/socher/2015input.txt")){
      for(ParaphraseProblem p:problems){
        w.println(p.label.isTrue()?1:0);
        w.println(p.s1.surface);
        w.println(p.s2.surface);
      }
    }
  }
  
  
  static void cacheAllAnnotation() throws IOException{
    List<ParaphraseProblem> problems = ParaphraseProblem.readAllProblems();
    Timing t = new Timing();
    t.start();
    for(ParaphraseProblem problem: problems){
      if(problem.s1.getDocAnnotation()==null 
          || problem.s2.getDocAnnotation()==null){
        System.err.println("null annotation?");
      }
      problem.s1.dispose();
      problem.s2.dispose();
    }
    t.done();
    t.report("Finished annotation caching ");
  }
  
  static void markShingles(String input,String output) throws IOException{
    
    int writeBuffer = 2<<20;
    
    Set<String> semevalShingles = 
        dumpShingles(null)
        .keySet()
        .stream()
        .map(String::toLowerCase)
        .collect(Collectors.toSet());
    
    LinkedList<String> window = Lists.newLinkedList();
    
    try(Writer writer = new BufferedWriter(new FileWriter(output), writeBuffer)){
      try(Scanner in = new Scanner(new File(input))){
        in.useDelimiter("\\s+");
        while(in.hasNext()){
          window.add(in.next());
          if(window.size()==3){
            
            String bigram = window.get(0) + "_" + window.get(1);
            String trigram = bigram + "_" + window.get(2);
            
            if(semevalShingles.contains(trigram.toLowerCase())){
              writer.write(trigram);
              window.clear();
            }else{
              if(semevalShingles.contains(bigram.toLowerCase())){
                writer.write(bigram);
                window.removeFirst();
                window.removeFirst();
              }else{
                // now nothing can be combined for a phrase with this word
                // write the word and continue
                writer.write(window.removeFirst());
              }
            }
            writer.write(" ");
          }
        }// end while
        
        // flush the remaining segments
        writer.write(StringUtils.join(window," "));
        
      }//end scan
    }
    
  }
  
  

  @FunctionalInterface
  private interface ThrowableFunction<T, R> {
    R apply(T a) throws Exception;
  }

  @FunctionalInterface
  private interface ThrowableConsumer<T> {
    void accept(T a) throws Exception;
  }

  public static void printMatlabMatrix(double[][] matrix,PrintStream out) {
    
    out.print("[");
    boolean first = true;
    for(double[] row: matrix){
      if(first){
        first = false;
      }else
        out.print(";");
      out.print(Doubles.join(" ", row));
    }
    out.println("]");
    
  }

}
