#why slim because its a lightweight python image
FROM python:3.14-slim

#set the working directory inside container
WORKDIR /app

#Copy dependency list before copying application code
COPY requirements.txt .

#install the dependencies
RUN pip install -r requirements.txt

#Copy application code and saved odel inside /app
COPY . .

#Document the port used by the fast api application
EXPOSE 8000

#start the api server when api is ready
#0.0.0.0 -allows connections through the container's network interface
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8000"]
