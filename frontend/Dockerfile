FROM node:22-alpine

WORKDIR /app

# Install dependencies first for layer caching
COPY package.json package-lock.json ./
RUN npm ci

# Copy config files (source is volume-mounted in dev)
COPY vite.config.ts tsconfig*.json index.html ./

# Copy source (overridden by volume mount in dev)
COPY src/ src/
COPY public/ public/

EXPOSE 5151

CMD ["npx", "vite", "--host", "0.0.0.0"]
