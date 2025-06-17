# 1. Stop all running containers
docker stop $(docker ps -aq)

# 2. Remove all containers
docker rm $(docker ps -aq)

# 3. Remove all images
docker rmi $(docker images -q)

# 4. Remove all volumes
docker volume rm $(docker volume ls -q)

# 5. Remove all custom networks (leaving bridge, host, none)
docker network rm $(docker network ls --filter "type=custom" -q)

# 6. Prune any leftover build cache, dangling data, and unused objects
docker system prune -a --volumes -f

# 7. (Optional) Uninstall Docker packages
#apt purge -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# 8. (Optional) Remove all Docker directories
#rm -rf /var/lib/docker /var/lib/containerd /etc/docker
