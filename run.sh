export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/credentials/Yelp-Spam-Detection-fa1ca54489b4.json";
mvn clean package assemply:single;
java -jar target/document-ranking-1.0-jar-with-dependencies.jar
