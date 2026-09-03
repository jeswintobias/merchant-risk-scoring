FROM node:20-slim as build

WORKDIR /app

# Install dependencies
COPY app/frontend/package*.json ./
RUN npm install

# Copy source and build
COPY app/frontend/ ./
RUN npm run build

# Production stage using lightweight web server
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html

# Expose port
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
