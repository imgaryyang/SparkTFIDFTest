rm -rf outfile/; 
mvn package; 
spark-submit --class DocumentRanking --master local target/document-ranking-1.0.jar ;

