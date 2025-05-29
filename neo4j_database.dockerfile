FROM neo4j:5.18.0
COPY ./neo4j_data_for_image/data /data
RUN chown -R neo4j:neo4j /data
EXPOSE 7474 7687