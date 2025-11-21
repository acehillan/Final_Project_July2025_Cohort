-- AI Tutor Application Schema (MySQL)

-- 1. Users Table (To store student accounts)
-- We track grade level and a Kenyan location default for context.
DROP TABLE IF EXISTS users;
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL, -- Storing hashed password is crucial for security
    grade_level INT NOT NULL, -- Junior school grades (e.g., 4, 5, 6)
    location VARCHAR(100) DEFAULT 'Kenya',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Chat History Table (To store conversations for context)
-- This allows the AI to remember the previous questions in a session.
DROP TABLE IF EXISTS chat_history;
CREATE TABLE chat_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    session_id VARCHAR(255) NOT NULL, -- Groups messages into separate sessions
    message_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    role ENUM('user', 'model') NOT NULL, -- 'user' (student) or 'model' (AI tutor)
    content TEXT NOT NULL,
    -- Links chat history back to the user
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- You would need to re-run this updated schema against your MySQL database.