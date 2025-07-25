from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import logging
from nltk.corpus import stopwords
import nltk
from rapidfuzz import fuzz
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# SMTP Configuration
SMTP_SERVER = 'smtp.privateemail.com'
SMTP_PORT = 465
SMTP_USERNAME = 'info@stellsync.com'
SMTP_PASSWORD = 'StellSync@2025'  # Update if needed

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt_tab')
    nltk.download('stopwords')

app = Flask(__name__)
CORS(app)

# Enable error logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# FAQ data (91 entries, updated services and projects entries)
faq_data = [
    # Greetings
    {
        "question": "hi|hello|hey|greetings",
        "answer": "Hello! Welcome to StellSync Solutions. How can I assist you today?",
        "category": "greeting",
        "keywords": ["hi", "hello", "greet"]
    },
    {
        "question": "good morning|morning",
        "answer": "Good morning! How can I help you with your tech needs?",
        "category": "greeting",
        "keywords": ["morning"]
    },
    {
        "question": "good evening|evening",
        "answer": "Good evening! Ready to explore our services?",
        "category": "greeting",
        "keywords": ["evening"]
    },
    {
        "question": "thank you|thanks",
        "answer": "You're welcome! Anything else I can help with?",
        "category": "polite",
        "keywords": ["thank", "thanks"]
    },
    {
        "question": "bye|goodbye",
        "answer": "Goodbye! Feel free to reach out anytime at info@stellsync.com or +94 71 460 0333.",
        "category": "polite",
        "keywords": ["bye", "goodbye"]
    },
    # Company Info
    {
        "question": "about stellsync|tell me about your company|who are you|what is stellsync",
        "answer": "StellSync Solutions, founded in 2022 in Sri Lanka, is a premier software development and data science firm. With a team of 10 experts, we've completed over 20 projects across retail, energy, education, and government sectors.",
        "category": "company",
        "keywords": ["about", "company", "stellsync", "who"]
    },
    {
        "question": "where are you located|address|location",
        "answer": "We are based at No. 123, Tech Street, Colombo 05, Sri Lanka.",
        "category": "company",
        "keywords": ["location", "address", "where"]
    },
    {
        "question": "vision|what is your vision",
        "answer": "Our vision is to be Sri Lanka's most innovative technology partner, delivering intelligent solutions that transform businesses across South Asia.",
        "category": "company",
        "keywords": ["vision"]
    },
    {
        "question": "mission|what is your mission",
        "answer": "Our mission is to empower organizations with cutting-edge software and data solutions that solve real-world challenges and enhance efficiency.",
        "category": "company",
        "keywords": ["mission"]
    },
    {
        "question": "how long in business|when founded|years of operation",
        "answer": "StellSync Solutions was founded in 2022, with 3 years of experience delivering innovative solutions.",
        "category": "company",
        "keywords": ["founded", "years", "business"]
    },
    {
        "question": "company size|how many employees|team size",
        "answer": "We have a dedicated team of 10 highly skilled engineers and developers specializing in software and data science.",
        "category": "company",
        "keywords": ["team", "employees", "size"]
    },
    {
        "question": "client types|who are your clients",
        "answer": "Our clients include startups, SMEs, and large enterprises in retail, energy, education, and government sectors.",
        "category": "company",
        "keywords": ["clients", "who"]
    },
    {
        "question": "certifications|awards|accreditations",
        "answer": "We are ISO 9001:2015 certified for quality management and have received the Sri Lanka Innovation Award in 2024.",
        "category": "company",
        "keywords": ["certifications", "awards"]
    },
    {
        "question": "industries served|sectors you work in",
        "answer": "We serve retail, energy, education, government, healthcare, and logistics industries with tailored solutions.",
        "category": "company",
        "keywords": ["industries", "sectors"]
    },
    {
        "question": "office hours|working hours|when are you open|what time are you open|opening hours|business hours|what time do you open|when do you open|when are you available",
        "answer": "Our office hours are Monday to Friday, 9 AM to 5 PM (Sri Lanka Time). Support is available 24/7 via email at info@stellsync.com or call +94 71 460 0333.",
        "category": "contact",
        "keywords": ["hours", "open", "available"]
    },
    # Contact
    {
        "question": "contact|how to reach you|how to contact|contact details|contact information|stellsync contact|contact stellsync|stellsync solutions contact",
        "answer": "Reach us at info@stellsync.com or call +94 71 460 0333.",
        "category": "contact",
        "keywords": ["contact", "reach", "details", "information", "stellsync", "how to contact"]
    },
    {
        "question": "email address|email",
        "answer": "Reach us at info@stellsync.com or call +94 71 460 0333.",
        "category": "contact",
        "keywords": ["email"]
    },
    {
        "question": "phone number|call you|contact number",
        "answer": "Reach us at info@stellsync.com or call +94 71 460 0333.",
        "category": "contact",
        "keywords": ["phone", "call", "number"]
    },
    {
        "question": "visit office|can i visit",
        "answer": "Yes, visit us at No. 123, Tech Street, Colombo 05, Sri Lanka. Please schedule an appointment via info@stellsync.com or call +94 71 460 0333.",
        "category": "contact",
        "keywords": ["visit", "office"]
    },
    # Social Media
    {
        "question": "social media|social links|facebook|linkedin|github|tiktok|whatsapp|instagram",
        "answer": "Connect with us on social media:\n- LinkedIn: https://www.linkedin.com/in/stellsync-solution/\n- Facebook: https://www.facebook.com/profile.php?id=61578301712677\n- GitHub: https://github.com/StellSync\n- TikTok: https://www.tiktok.com/@stellsyncsolution\n- WhatsApp: https://whatsapp.com/channel/0029VbBnMlZHbFV7VIkmOm10\n- Instagram: https://www.instagram.com/stellsync",
        "category": "contact",
        "keywords": ["social", "facebook", "linkedin", "github", "tiktok", "whatsapp", "instagram"]
    },
    # Services
    {
        "question": "services|what do you offer|what services|stellsync services|services stellsync|what stellsync provided|what stellsync solution provided|what stellsync solutions provided",
        "answer": "We offer custom software development, AI/ML solutions, data science and analytics, mobile app development, cloud solutions, IoT, big data processing, DevOps, database administration, and system integration.",
        "category": "services",
        "keywords": ["services", "offer", "stellsync", "provided"]
    },
    {
        "question": "software development|custom software|app development",
        "answer": "We build tailored applications using React, Angular, Node.js, .NET Core, Django, and Spring Boot, designed for performance and scalability.",
        "category": "services",
        "keywords": ["software", "development", "app"]
    },
    {
        "question": "mobile apps|mobile app development|build apps",
        "answer": "We develop cross-platform apps using Flutter and React Native, as well as native apps for iOS (Swift) and Android (Kotlin).",
        "category": "services",
        "keywords": ["mobile", "apps"]
    },
    {
        "question": "data science|analytics|data analysis",
        "answer": "Our data science services include predictive modeling, business intelligence dashboards, and analytics using Python, R, SQL, Tableau, and Power BI.",
        "category": "services",
        "keywords": ["data", "analytics", "science"]
    },
    {
        "question": "machine learning|ai|artificial intelligence",
        "answer": "We provide AI solutions including NLP, computer vision, predictive analytics, recommendation systems, and chatbots using BERT, LLAMA, TensorFlow, and PyTorch.",
        "category": "services",
        "keywords": ["ai", "machine learning", "artificial intelligence"]
    },
    {
        "question": "cloud|cloud computing|cloud solutions",
        "answer": "We offer cloud architecture, migration, and optimization on AWS, Azure, and Google Cloud, including serverless and containerized solutions.",
        "category": "services",
        "keywords": ["cloud", "computing"]
    },
    {
        "question": "iot|internet of things|iot projects",
        "answer": "We integrate IoT devices with big data platforms for real-time monitoring and analytics, using MQTT, ESP32, and AWS IoT Core.",
        "category": "services",
        "keywords": ["iot", "internet of things"]
    },
    {
        "question": "data engineering|data warehouse|data lakehouse",
        "answer": "We provide data engineering solutions with Azure Data Factory, Snowflake, Microsoft Fabric, and lakehouses for efficient data pipelines.",
        "category": "services",
        "keywords": ["data engineering", "warehouse", "lakehouse"]
    },
    {
        "question": "devops|ci/cd|continuous integration",
        "answer": "Our DevOps services include CI/CD pipelines, containerization with Docker and Kubernetes, and infrastructure as code with Terraform.",
        "category": "services",
        "keywords": ["devops", "ci/cd"]
    },
    {
        "question": "big data|big data processing",
        "answer": "We handle large-scale data processing with Hadoop, Spark, and Kafka for real-time and batch analytics.",
        "category": "services",
        "keywords": ["big data", "processing"]
    },
    {
        "question": "database|oracle|sql|database administration",
        "answer": "We offer database administration and optimization for Oracle, MySQL, PostgreSQL, MongoDB, and cloud-native databases.",
        "category": "services",
        "keywords": ["database", "sql", "administration"]
    },
    {
        "question": "chatbot|build chatbots|chatbot development",
        "answer": "We create rule-based and AI-powered chatbots using frameworks like Rasa, Dialogflow, and custom LLAMA models.",
        "category": "services",
        "keywords": ["chatbot", "development"]
    },
    {
        "question": "optimization|genetic algorithm|reinforcement learning|pso",
        "answer": "We develop optimization solutions using Genetic Algorithms, Reinforcement Learning, Particle Swarm Optimization, and simulated annealing.",
        "category": "services",
        "keywords": ["optimization", "genetic algorithm", "reinforcement learning"]
    },
    {
        "question": "time series|forecasting|time series forecasting",
        "answer": "Our forecasting solutions use LSTM, ARIMA, and Prophet for accurate predictions in demand and resource planning.",
        "category": "services",
        "keywords": ["time series", "forecasting"]
    },
    {
        "question": "nlp|natural language processing",
        "answer": "We offer NLP solutions for text analysis, sentiment detection, entity recognition, and chatbot development.",
        "category": "services",
        "keywords": ["nlp", "natural language processing"]
    },
    {
        "question": "recommendation system|recommender system",
        "answer": "We build recommendation systems using collaborative filtering, content-based filtering, and deep learning models.",
        "category": "services",
        "keywords": ["recommendation", "recommender"]
    },
    {
        "question": "web development|website development",
        "answer": "We develop responsive websites with React, Angular, Vue.js, Django, Flask, and Node.js.",
        "category": "services",
        "keywords": ["web", "website", "development"]
    },
    {
        "question": "testing|quality assurance|qa",
        "answer": "We provide QA services including automated testing with Selenium, unit testing, and performance testing.",
        "category": "services",
        "keywords": ["testing", "qa", "quality assurance"]
    },
    {
        "question": "security|cybersecurity|data protection",
        "answer": "We implement robust security measures including encryption, secure APIs, and compliance with GDPR and ISO standards.",
        "category": "services",
        "keywords": ["security", "cybersecurity"]
    },
    {
        "question": "maintenance|support|post-development",
        "answer": "We offer 24/7 maintenance and support, including bug fixes, updates, and monitoring.",
        "category": "services",
        "keywords": ["maintenance", "support"]
    },
    # Projects and Solutions
    {
        "question": "projects|case studies|work samples|featured work",
        "answer": (
            "We have delivered impactful solutions:\n"
            "- Dockerized Machine Learning Data Server: Python server streaming Iris dataset via TCP, containerized with Docker.\n"
            "- Supply Chain Optimization: Flask app using PuLP for cost optimization with interactive visualizations.\n"
            "- IoT Environmental Monitoring: Real-time platform with ESP32, Node-RED, MQTT, and LSTM forecasting.\n"
            "- Water Quality Prediction: ML tool using KNN, Decision Tree, Naive Bayes, and XGBoost, deployed as a Flask app.\n"
            "- Microservices-Based Learning Platform: Scalable LMS using Docker and Spring Boot.\n"
            "- Inteliguide AI Tourist Assistant: Personalized travel recommendations via Kotlin Android app.\n"
            "- FitConnect Social Fitness Network: Social platform with Spring Boot, Next.js, MySQL, Docker, and AWS.\n"
            "- CloudServe Scalable Web App: Cloud-native app using AWS EC2, S3, RDS, and Lambda.\n"
            "- AzureSphere Enterprise Platform: Azure-based solution with App Service, Functions, SQL, and DevOps.\n"
            "- VisionBlend Shopping App: Android app for visually impaired with voice assistance and dynamic themes.\n"
            "- EZLiving Inclusive E-Commerce: MERN platform with voice control for accessibility.\n"
            "- EliteWear E-Commerce: Multi-platform solution with .NET, React, and Kotlin.\n"
            "- VegiTrace Farmer & Vendor App: Location-aware app with GPS and QR verification.\n"
            "- Edu Wave E-Learning Platform: Microservices-based LMS with MERN and Docker.\n"
            "- Wildlife Safari Trip Management: PHP-based web portal for safari bookings.\n"
            "- UX/UI Website Redesign: Figma prototype for improved usability and accessibility.\n"
            "Let me know if you’d like details about any of these."
        ),
        "category": "projects",
        "keywords": ["projects", "portfolio", "case studies"]
    },
    {
        "question": "client feedback|testimonials|reviews",
        "answer": "Our clients appreciate our focus on results and collaboration. For example: 'StellSync's forecasting platform transformed our inventory management and improved profitability.'",
        "category": "projects",
        "keywords": ["feedback", "testimonials", "reviews"]
    },
    # Individual Project FAQs
    {
        "question": "dockerized machine learning|ml data server",
        "answer": "Our Dockerized Machine Learning Data Server is a Python application streaming the Iris dataset to clients over TCP, containerized with Docker for portability and easy deployment.",
        "category": "projects",
        "keywords": ["dockerized", "machine learning", "data server"]
    },
    {
        "question": "supply chain optimization|supply chain",
        "answer": "The Supply Chain Optimization project is a Flask web app using Linear Programming (PuLP) to optimize manufacturing and freight costs, featuring interactive visualizations.",
        "category": "projects",
        "keywords": ["supply chain", "optimization"]
    },
    {
        "question": "iot environmental monitoring|iot monitoring",
        "answer": "Our IoT Environmental Monitoring platform for smart school zones uses ESP32 sensors, Node-RED, MQTT, and LSTM forecasting, with live dashboards and Telegram alerts.",
        "category": "projects",
        "keywords": ["iot", "environmental", "monitoring"]
    },
    {
        "question": "water quality prediction|water quality",
        "answer": "The Water Quality Prediction tool uses ML models (KNN, Decision Tree, Naive Bayes, XGBoost) developed in Google Colab and deployed as a Flask web app.",
        "category": "projects",
        "keywords": ["water quality", "prediction"]
    },
    {
        "question": "microservices learning platform|lms",
        "answer": "Our Microservices-Based Learning Platform is a modular LMS using Docker and Spring Boot for scalable course management and independent deployment.",
        "category": "projects",
        "keywords": ["microservices", "learning platform", "lms"]
    },
    {
        "question": "inteliguide|tourist assistant",
        "answer": "Inteliguide is an AI-powered Android app built with Kotlin, recommending personalized tourist destinations based on user preferences and real-time data.",
        "category": "projects",
        "keywords": ["inteliguide", "tourist", "assistant"]
    },
    {
        "question": "fitconnect|fitness network",
        "answer": "FitConnect is a social fitness platform built with Spring Boot, Next.js, MySQL, Docker, and AWS, enabling workout sharing and community engagement.",
        "category": "projects",
        "keywords": ["fitconnect", "fitness", "social"]
    },
    {
        "question": "cloudserve|aws web app",
        "answer": "CloudServe is a cloud-native web app leveraging AWS EC2, S3, RDS, and Lambda for high availability, auto-scaling, and secure operations.",
        "category": "projects",
        "keywords": ["cloudserve", "aws", "web app"]
    },
    {
        "question": "azuresphere|azure platform",
        "answer": "AzureSphere is an enterprise cloud platform using Azure App Service, Functions, SQL, and DevOps for scalable compute and analytics.",
        "category": "projects",
        "keywords": ["azuresphere", "azure", "enterprise"]
    },
    {
        "question": "visionblend|shopping app for visually impaired",
        "answer": "VisionBlend is an Android shopping app built with Kotlin and Firebase, offering voice assistance, dynamic themes, and magnifying effects for visually impaired users.",
        "category": "projects",
        "keywords": ["visionblend", "visually impaired", "shopping"]
    },
    {
        "question": "ezliving|e-commerce accessibility",
        "answer": "EZLiving is an accessible MERN e-commerce platform with voice control and customizable themes, designed for visually impaired users.",
        "category": "projects",
        "keywords": ["ezliving", "e-commerce", "accessibility"]
    },
    {
        "question": "elitewear|e-commerce platform",
        "answer": "EliteWear is a multi-platform e-commerce solution with a .NET backend, React frontend, and Kotlin Android app, featuring vendor filtering and secure authentication.",
        "category": "projects",
        "keywords": ["elitewear", "e-commerce"]
    },
    {
        "question": "vegitrace|farmer vendor app",
        "answer": "VegiTrace is a location-aware Android app built with Kotlin, featuring GPS tracking and QR-based vegetable batch verification for agricultural logistics.",
        "category": "projects",
        "keywords": ["vegitrace", "farmer", "vendor"]
    },
    {
        "question": "edu wave|e-learning platform",
        "answer": "Edu Wave is a web-based LMS with a microservices architecture using MERN Stack and Docker, supporting scalable course management and payments.",
        "category": "projects",
        "keywords": ["edu wave", "e-learning", "lms"]
    },
    {
        "question": "wildlife safari|safari management",
        "answer": "The Wildlife Safari Trip Management System is a PHP-based web portal for managing safari trips, bookings, and customer inquiries, with About Us and Contact Us pages.",
        "category": "projects",
        "keywords": ["wildlife", "safari", "management"]
    },
    {
        "question": "ux ui redesign|website redesign",
        "answer": "Our UX/UI Website Redesign project is a Figma prototype focusing on improved usability, accessibility, and visual design consistency.",
        "category": "projects",
        "keywords": ["ux", "ui", "redesign"]
    },
    # HR / Careers
    {
        "question": "jobs|careers|vacancies|job openings|positions available|open positions|hiring now|how to find a job|how can i find a job|find a job|job opportunities at stellsync",
        "answer": "We’re always looking for talented individuals. View openings at www.stellsync.com/careers or email hr@stellsync.com",
        "category": "hr",
        "keywords": ["jobs", "careers", "vacancies", "opportunities"]
    },
    {
        "question": "apply for a job|how can i apply|submit application|send my cv|start a job|how can i start a job|join your team|get hired",
        "answer": "To apply, please email your resume to hr@stellsync.com with a short introduction about yourself.",
        "category": "hr",
        "keywords": ["apply", "job", "cv"]
    },
    {
        "question": "salary|compensation|remuneration|wages",
        "answer": "Salary varies by role and experience. We offer competitive compensation aligned with industry standards.",
        "category": "hr",
        "keywords": ["salary", "compensation"]
    },
    {
        "question": "working hours|work schedule|work time|office hours|shift timings",
        "answer": "Our regular working hours are Monday to Friday, 9 AM to 5 PM. Flexible and hybrid options are available.",
        "category": "hr",
        "keywords": ["working hours", "schedule"]
    },
    {
        "question": "internship|graduate programs|trainee positions|student opportunities",
        "answer": "Yes, we offer internships and graduate programs. Email hr@stellsync.com for details.",
        "category": "hr",
        "keywords": ["internship", "graduate"]
    },
    {
        "question": "work culture|company culture|team atmosphere",
        "answer": "We foster a collaborative, inclusive environment that encourages innovation and work-life balance.",
        "category": "hr",
        "keywords": ["culture", "work environment"]
    },
    {
        "question": "benefits|perks|employee benefits|incentives",
        "answer": "Benefits include health insurance, development programs, bonuses, and flexible work.",
        "category": "hr",
        "keywords": ["benefits", "perks"]
    },
    {
        "question": "areas hiring|departments hiring|roles available",
        "answer": "We frequently hire in software development, data science, project management, and DevOps.",
        "category": "hr",
        "keywords": ["hiring", "roles"]
    },
    # Technical
    {
        "question": "architecture|software architecture|design patterns",
        "answer": "We use microservices, MVC, and event-driven architectures, with patterns like Singleton and Factory.",
        "category": "technical",
        "keywords": ["architecture", "design patterns"]
    },
    {
        "question": "performance|optimization|scalability",
        "answer": "We optimize performance with load balancing, caching, and database indexing.",
        "category": "technical",
        "keywords": ["performance", "scalability"]
    },
    {
        "question": "tech stack|technologies used|tools",
        "answer": "Our tech stack includes Python, JavaScript, Java, C#, React, Angular, Node.js, Django, Flask, .NET Core, AWS, Azure, Docker, Kubernetes, and more.",
        "category": "technical",
        "keywords": ["tech stack", "technologies"]
    },
    {
        "question": "agile|methodology|development process",
        "answer": "We follow Agile (Scrum/Kanban), with iterative development and client collaboration.",
        "category": "technical",
        "keywords": ["agile", "methodology"]
    },
    {
        "question": "apis|api development|integration",
        "answer": "We develop RESTful and GraphQL APIs and integrate with third-party services.",
        "category": "technical",
        "keywords": ["apis", "integration"]
    },
    {
        "question": "deployment|hosting|servers",
        "answer": "We deploy on AWS, Azure, and GCP with CI/CD pipelines.",
        "category": "technical",
        "keywords": ["deployment", "hosting"]
    },
    {
        "question": "machine learning models|ml models|ai models",
        "answer": "We use LLAMA, BERT, TensorFlow, PyTorch for NLP, computer vision, and analytics.",
        "category": "technical",
        "keywords": ["machine learning", "models"]
    },
    {
        "question": "data pipelines|etl|data processing",
        "answer": "We build ETL pipelines with Apache Airflow, Azure Data Factory, and Talend.",
        "category": "technical",
        "keywords": ["data pipelines", "etl"]
    },
    # Payment and Billing
    {
        "question": "discounts|special offers|promotions|deals",
        "answer": (
            "Yes! We periodically offer special discounts and promotions for new and returning clients. "
            "Please email sales@stellsync.com or visit our website for current offers and eligibility."
        ),
        "category": "billing",
        "keywords": ["discounts", "promotions"]
    },
    {
        "question": (
            "installments|instalments|payment plan|pay in parts|split payment|pay in instalments|"
            "can I pay in instalments|can I pay by instalments|pay over time|flexible payment"
        ),
        "answer": (
            "We understand flexibility is important. Depending on the project size and duration, "
            "we can arrange milestone-based payments or instalment plans. Contact sales@stellsync.com to discuss options."
        ),
        "category": "billing",
        "keywords": ["installments", "payment plan"]
    },
    {
        "question": (
            "payment methods|how to pay|card or cash|credit card|debit card|bank transfer|cash payment|"
            "payment options|i want to pay by card|i will pay by card|i like to pay by card|"
            "i like to pay in card|i prefer card payment|can i pay with card|pay using card|pay by card"
        ),
        "answer": (
            "We accept payments via bank transfer, credit/debit cards, and online payment gateways. "
            "For larger projects, milestone payments are recommended. Let us know your preferred method!"
        ),
        "category": "billing",
        "keywords": ["payment methods", "card"]
    },
    {
        "question": (
            "how to make a payment|how do i pay for a project|how to do a payment for a project|"
            "how to finalize payment|how do i complete payment|payment process|pay my invoice"
        ),
        "answer": (
            "To make a payment:\n"
            "1️⃣ Finalize your project details and agreement with our development team.\n"
            "2️⃣ We will issue an invoice with the payment amount and bank details.\n"
            "3️⃣ Make your advance or milestone payment via bank transfer, credit/debit card, or online gateway.\n"
            "4️⃣ Send the payment slip or confirmation to billing@stellsync.com or WhatsApp it to +94 71 460 0333.\n"
            "5️⃣ We will confirm receipt and proceed with development.\n\n"
            "If you have any questions, feel free to contact our billing team!"
        ),
        "category": "billing",
        "keywords": ["payment process", "invoice"]
    },
    {
        "question": "invoices|billing|invoice details|billing information|invoice copy|get invoice|request invoice",
        "answer": (
            "Invoices are issued for each project milestone or monthly services. "
            "If you need a copy or have questions about billing, please email billing@stellsync.com."
        ),
        "category": "billing",
        "keywords": ["invoices", "billing"]
    },
    {
        "question": "refunds|cancellation|money back|cancel contract|cancel my project|get refund",
        "answer": (
            "Our contracts include terms for cancellation and refunds, depending on the project stage and work completed. "
            "Contact us to review your agreement details."
        ),
        "category": "billing",
        "keywords": ["refunds", "cancellation"]
    },
    {
        "question": "payment terms|due dates|payment deadline|when do I pay|when is payment due",
        "answer": (
            "Payment terms are usually milestone-based or monthly, with due dates specified in your agreement. "
            "We’ll always provide clear timelines and reminders before payments are due."
        ),
        "category": "billing",
        "keywords": ["payment terms", "due dates"]
    },
    {
        "question": "receipt|payment confirmation|proof of payment",
        "answer": (
            "After each payment, we issue a receipt and confirmation. If you need another copy, email billing@stellsync.com."
        ),
        "category": "billing",
        "keywords": ["receipt", "confirmation"]
    },
    # Client and Project Process
    {
        "question": "start a project|how to start|new project",
        "answer": "Contact info@stellsync.com to discuss your requirements. We'll provide a proposal and timeline.",
        "category": "process",
        "keywords": ["start", "project"]
    },
    {
        "question": "nda|non-disclosure|confidentiality",
        "answer": "Yes, we sign NDAs to protect your confidentiality.",
        "category": "process",
        "keywords": ["nda", "confidentiality"]
    },
    {
        "question": "development time|how long|project timeline",
        "answer": "Project timelines vary. Small projects take 1-3 months, complex ones 6-12 months.",
        "category": "process",
        "keywords": ["timeline", "development time"]
    },
    {
        "question": "cost|pricing|how much",
        "answer": "Costs depend on complexity. Email info@stellsync.com for a customized quote.",
        "category": "process",
        "keywords": ["cost", "pricing"]
    },
    {
        "question": "project management|how you manage projects",
        "answer": "We use Agile management with Jira/Trello and provide regular updates.",
        "category": "process",
        "keywords": ["project management"]
    },
    {
        "question": "client collaboration|how you work with clients",
        "answer": "We collaborate closely via meetings, reports, and feedback sessions.",
        "category": "process",
        "keywords": ["collaboration", "clients"]
    },
    {
        "question": "post-delivery|after project|support",
        "answer": "We offer post-delivery support, maintenance, updates, and 24/7 assistance.",
        "category": "process",
        "keywords": ["post-delivery", "support"]
    },
    # Combined
    {
        "question": "projects and services|services and projects|what are you doing|what services you are giving|what services you provide|what projects you are doing|current projects|what project are you doing|tell me about your projects and services",
        "answer": (
            "We offer a wide range of services including:\n"
            "- Custom Software Development\n"
            "- AI/ML Solutions\n"
            "- Data Science & Analytics\n"
            "- Mobile App Development\n"
            "- Cloud Solutions\n"
            "- IoT & Big Data\n"
            "- DevOps & System Integration\n\n"
            "We have also delivered impactful solutions such as:\n"
            "- Dockerized Machine Learning Data Server\n"
            "- Supply Chain Optimization\n"
            "- IoT Environmental Monitoring\n"
            "- Water Quality Prediction\n"
            "- Microservices-Based Learning Platform\n"
            "- Inteliguide AI Tourist Assistant\n"
            "- FitConnect Social Fitness Network\n"
            "- CloudServe Scalable Web App\n"
            "- AzureSphere Enterprise Platform\n"
            "- VisionBlend Shopping App\n"
            "- EZLiving Inclusive E-Commerce\n"
            "- EliteWear E-Commerce\n"
            "- VegiTrace Farmer & Vendor App\n"
            "- Edu Wave E-Learning Platform\n"
            "- Wildlife Safari Trip Management\n"
            "- UX/UI Website Redesign\n\n"
            "Let me know if you’d like more details about any of these."
        ),
        "category": "combined",
        "keywords": ["projects", "services"]
    }
]

# Preprocess text for better matching
stop_words = set(stopwords.words('english'))
from spellchecker import SpellChecker
spell = SpellChecker()

def preprocess_text(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    # Normalize "stellsync solutions" and "stellsync solution" to "stellsync"
    text = re.sub(r"\bstellsync\s+solutions?\b", "stellsync", text)
    tokens = text.split()
    corrected = [spell.correction(word) for word in tokens]
    safe_corrected = [c if c is not None else w for c, w in zip(corrected, tokens)]
    tokens = [t for t in safe_corrected if t not in stop_words]
    return " ".join(tokens)

# Prepare questions for vectorization
questions = [item["question"] for item in faq_data]
processed_questions = [preprocess_text(q.split('|')[0]) for q in questions]

# Initialize TF-IDF vectorizer with n-grams
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95)
question_vectors = vectorizer.fit_transform(processed_questions)

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        if not data or "message" not in data:
            logger.warning("Invalid request: Missing 'message' field")
            return jsonify({"response": "Invalid request. Please send a JSON with a 'message' field."}), 400
        user_message = data.get("message", "").strip()
        if not user_message:
            logger.info("Empty message received")
            return jsonify({"response": "Please enter a message so I can assist you."}), 400

        # Preprocess user message
        processed_user_message = preprocess_text(user_message)

        # Check for out-of-scope questions
        out_of_scope_keywords = [
            "usa", "weather", "news", "politics", "sports", "movies", "music", "food",
            "travel", "health", "fashion", "celebrities", "stock market", "covid", "history"
        ]
        if any(keyword in user_message.lower() for keyword in out_of_scope_keywords):
            response = (
                "Sorry, that question is outside my knowledge base. I can help with information about "
                "StellSync Solutions, our services (e.g., software development, AI, cloud solutions), "
                "projects (e.g., VisionBlend, Edu Wave), or contact details. Please try a relevant question!"
            )
            logger.info(f"Out-of-scope question detected: '{user_message}'")
            return jsonify({"response": response})

        # Fuzzy matching with keyword prioritization
        best_score = 0
        best_item = None
        for item in faq_data:
            patterns = item["question"].split('|')
            for pattern in patterns:
                similarity = fuzz.token_sort_ratio(pattern.strip(), user_message.lower())
                # Boost score for keywords
                keyword_score = 0
                for keyword in item["keywords"]:
                    if keyword.lower() in user_message.lower():
                        keyword_score += 30 if "contact" in keyword.lower() else 20
                total_score = similarity + keyword_score
                # Penalize non-contact categories if "contact" is in the query
                if "contact" in user_message.lower() and item["category"] != "contact":
                    total_score *= 0.7
                # Boost services category if "what" and "provided" are in the query
                if "what" in user_message.lower() and "provided" in user_message.lower() and item["category"] == "services":
                    total_score += 20
                if total_score > best_score:
                    best_score = total_score
                    best_item = item
        if best_score >= 80:
            logger.info(f"Fuzzy best match: '{best_item['question']}' ({best_score}%)")
            return jsonify({"response": best_item["answer"]})

        # Fallback to TF-IDF similarity
        user_vector = vectorizer.transform([processed_user_message])
        similarities = cosine_similarity(user_vector, question_vectors).flatten()
        max_index = np.argmax(similarities)
        max_score = similarities[max_index]
        threshold = 0.3

        if max_score >= threshold:
            response = faq_data[max_index]["answer"]
            logger.info(f"Similarity match for '{user_message}': {response} (score: {max_score})")
            return jsonify({"response": response})

        # Generic fallback for no match
        response = (
            "I'm not sure I understand. Could you rephrase or ask about our services, like AI, software development, "
            "or our portfolio? Contact us at info@stellsync.com or call +94 71 460 0333 for more help."
        )
        logger.info(f"No match for '{user_message}' (score: {max_score})")
        return jsonify({"response": response})

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return jsonify({"response": "An error occurred. Please try again later."}), 500

@app.route('/send-message', methods=['POST'])
def send_message():
    try:
        data = request.json
        name = data.get('name', '')
        email = data.get('email', '')
        subject = data.get('subject', '')
        message = data.get('message', '')
        if not all([name, email, subject, message]):
            return jsonify({'status': 'error', 'message': 'All fields are required'}), 400

        # Compose email
        msg = MIMEMultipart()
        msg['From'] = SMTP_USERNAME
        msg['To'] = SMTP_USERNAME  # Sends to info@stellsync.com
        msg['Subject'] = f"[Contact Form] {subject}"
        body = f"""
You have received a new message from the StellSync website contact form:

📌 Name: {name}
📧 Email: {email}
📝 Subject: {subject}
💬 Message:
{message}
"""
        msg.attach(MIMEText(body, 'plain'))

        # Send via PrivateEmail with proper error handling
        try:
            with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
                server.login(SMTP_USERNAME, SMTP_PASSWORD)
                server.send_message(msg)
            logger.info(f"Email sent successfully to {msg['To']} from {msg['From']}")
            return jsonify({'status': 'success', 'message': 'Message sent successfully'})
        except smtplib.SMTPAuthenticationError as auth_error:
            logger.error(f"SMTP Authentication Error: {str(auth_error)}")
            return jsonify({'status': 'error', 'message': 'Authentication failed. Please check your SMTP credentials.'}), 500
        except Exception as e:
            logger.error(f"Error sending email: {str(e)}")
            return jsonify({'status': 'error', 'message': 'Failed to send message due to a server error.'}), 500
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({'status': 'error', 'message': 'Failed to process request'}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)

#comment
