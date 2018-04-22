import scala.Tuple2;

import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.*;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.SparkConf;

import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;
import java.util.Collection;
import java.util.Comparator;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public final class DocumentRanking {

  public static void main(String[] args) throws Exception {

    String appName = "document-ranking";
    String stopwordsFileName = "AssignmentData/stopwords.txt";
    String inputFilePath = "AssignmentData/datafiles";
    String queryFileName = "AssignmentData/query.txt";
    String outputTextFileName = "outfile/output.txt";
    Boolean isSortAscending = false;
    int k = 3;
    final int inputFilesCount = new File(inputFilePath).list().length;

    //read in the stopwords
    List<String> stopwordsList = new ArrayList<>();
    try (Stream<String> stream = Files.lines(Paths.get(stopwordsFileName))) {
        stopwordsList = stream.flatMap(v -> Arrays.stream(v.split("[ \t\n\f\r]")))
                                                  .collect(Collectors.toList());
    }
    catch (IOException e) {
	    e.printStackTrace();
    }
    final List<String> stopwords = stopwordsList;
    
    //read in the query
    List<String> queryList = new ArrayList<>();
    try (Stream<String> stream = Files.lines(Paths.get(queryFileName))) {
        queryList = stream.flatMap(v -> Arrays.stream(v.split("[ \t\n\f\r]")))
                                              .collect(Collectors.toList());
    }
    catch (IOException e) {
	    e.printStackTrace();
    }
    final List<String> query = queryList;

    //create Spark context with Spark configuration
    JavaSparkContext sc = new JavaSparkContext(new SparkConf().setAppName(appName)); 

    //set the input file
    JavaPairRDD<String, String> textFiles = sc.wholeTextFiles(inputFilePath);

    //word count inside each doc process
    JavaPairRDD<String, String> wordCountPerDoc = textFiles
        //create a list of {fileName}@{word} for all words in all files
    	.flatMap(s -> Arrays.asList(s._2()
    	                             //split text by all white space
			                         .split("[ \t\n\f\r]"))
			                .stream()
			                //remove stop words
			                .filter(token -> !stopwords.contains(token))
			                //convert to lower case
			                .map(String::toLowerCase)
			                //remove non-words
			                .map(rawWord -> rawWord.replaceAll("[^\\p{L}\\p{Nd}]+", ""))
			                //convert data to {fileName}@{word}
			                .map(w -> s._1()
				                       .substring(s._1()
				                   		           .lastIndexOf("/") + 1)
				                       .split("\\.")[0] + 
				                       "@" + 
				                       w)
			                .iterator())
	    //count how many words are in each file
    	.mapToPair(word -> new Tuple2<>(word, 1))
    	.reduceByKey((a, b) -> a + b)
    	//map key to the word and value to {fileName}={wordCountInFile}
        .mapToPair(input -> new Tuple2<>(input._1()
					                          .substring(input._1()
					       	  	                              .lastIndexOf("@") + 1),
					                     input._1()
						                      .split("@")[0] +
					                          "=" +
						                      input._2()));
	
	//calculate tf-idf per file process	
    JavaPairRDD<String, String> tfIdf = wordCountPerDoc
        //calculate documents per word
        .mapValues(val -> 1)
        .reduceByKey((a, b) -> a + b)
        //combine with the words per document count (resulting in Tuple2<docCount:Integer, wordCount:String>)
        .join(wordCountPerDoc)
        //calculate TF-IDF from the tuples Tuple2<word:String, Tuple2<docCount:Integer, wordCount:String>>
        .mapToPair(input -> {
                       String word = input._1();
                       Integer docCount = input._2()._1();
                       String[] wordCountString = input._2()._2().split("=");
                       String docName = wordCountString[0];
                       Integer wordCount = Integer.parseInt(wordCountString[1]);
                       Double tfIdfVal = (1 + Math.log(wordCount)) * Math.log(inputFilesCount / docCount);
                       String wordDocKey = word + "@" + docName;
                       return new Tuple2<>(wordDocKey, tfIdfVal);
                   })
        //map to word tf-idf for each document
        .mapToPair(input -> new Tuple2<>(input._1()
                                              .substring(input._1()
                                                              .lastIndexOf("@") + 1),
                                         input._1()
                                              .split("@")[0] +
                                              "=" +
                                              input._2()));
                   
    //calculate normalized tf-idf process
    JavaPairRDD<String, Double> normTfIdf = tfIdf
        //calculate the sum of square of TF-IDF for all words in the document
        .mapValues(val -> Double.parseDouble(val.substring(val.lastIndexOf("=") + 1)))
        .reduceByKey((accumulator, value) -> accumulator + (value * value))
        //combine with the tf-idf (resulting in Tuple2<sumOfSquare:Double, tfIdf:String>)
        .join(tfIdf)
        //calculate norm TF-IDF from the tuples Tuple2<docName:String, Tuple2<sumOfSquare:Double, tfIdf:String>>
        .mapToPair(input -> {
                      String docName = input._1();
                      Double docSumOfSquare = input._2()._1();
                      String[] tfIdfString = input._2()._2().split("=");
                      String word = tfIdfString[0];
                      Double tfIdfVal = Double.parseDouble(tfIdfString[1]);
                      Double normTfIdfVal = tfIdfVal / Math.sqrt(docSumOfSquare);
                      String normTfIdfKey = word + "@" + docName;
                      return new Tuple2<>(normTfIdfKey, normTfIdfVal);
                  });
                  
    //calculate the document relevance from query process
    JavaPairRDD<String, Double> relevance = normTfIdf
        //map to key to the document name and value to be the norm tf-idf
        .mapToPair(input -> new Tuple2<>(input._1()
                                              .substring(input._1()
                                                              .lastIndexOf("@") + 1),
                                         input._1()
                                              .split("@")[0] +
                                              "=" +
                                              input._2()))
        //filter out words that are not in the query
        .filter(input -> query.contains(input._2().split("=")[0]))
        .mapValues(val -> Double.parseDouble(val.substring(val.lastIndexOf("=") + 1)))
        //sum up the norm tf-idf for each document
        .reduceByKey((a, b) -> a + b);
        
    //sort the output in descending order and select the top k results  
    List<Tuple2<String, Double>> sortedList = relevance
        .takeOrdered(k, new TFIDFComparatorDesc());
	
    //set the output folder
    JavaPairRDD<String, Double> sorted = sc.parallelizePairs(sortedList);
    sorted.saveAsTextFile(outputTextFileName);
    //stop spark
  }
  
  static class TFIDFComparatorDesc implements Comparator<Tuple2<String, Double>>, Serializable {
      @Override
      public int compare(Tuple2<String, Double> t1, Tuple2<String, Double> t2) {
          return -t1._2().compareTo(t2._2());
      }
  }
}
