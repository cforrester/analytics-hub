docker build -t upa-engine:latest .
docker rm -f upa_engine_container      # stop & remove old container
docker run -d --name upa_engine_container -p 8501:8501 upa-engine:latest
