FROM opensearchproject/opensearch:2.9.0

ENV discovery.type=single-node

EXPOSE 9200

CMD ["opensearch"]
