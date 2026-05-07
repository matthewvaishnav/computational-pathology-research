# SQL Injection Protection Guide

## Overview
This document outlines SQL injection prevention practices for the HistoCore framework.

## Rules

### ✅ ALWAYS Use Parameterized Queries

```python
# CORRECT - Parameterized query
cursor.execute("SELECT * FROM users WHERE username = ?", (username,))

# CORRECT - Named parameters
cursor.execute("SELECT * FROM users WHERE username = :username", {"username": username})

# CORRECT - SQLAlchemy ORM
session.query(User).filter(User.username == username).first()
```

### ❌ NEVER Use String Formatting

```python
# WRONG - SQL injection vulnerability
cursor.execute(f"SELECT * FROM users WHERE username = '{username}'")

# WRONG - String concatenation
cursor.execute("SELECT * FROM users WHERE username = '" + username + "'")

# WRONG - % formatting
cursor.execute("SELECT * FROM users WHERE username = '%s'" % username)
```

## Validation

All database queries must:
1. Use parameterized queries or ORM
2. Never concatenate user input into SQL strings
3. Validate input types before queries
4. Use prepared statements for repeated queries

## Audit

Run this command to check for SQL injection vulnerabilities:

```bash
grep -r "execute.*%" src/
grep -r "execute.*\+" src/
grep -r "execute.*format" src/
```

## Approved Patterns

### SQLite with Parameters
```python
conn.execute("INSERT INTO table (col) VALUES (?)", (value,))
```

### PostgreSQL with psycopg2
```python
cursor.execute("SELECT * FROM table WHERE id = %s", (id,))
```

### SQLAlchemy ORM
```python
session.query(Model).filter(Model.field == value).all()
```

### SQLAlchemy Core
```python
stmt = select(table).where(table.c.field == bindparam('value'))
conn.execute(stmt, {'value': user_input})
```

## Testing

All database code must include tests for:
- SQL injection attempts (e.g., `' OR '1'='1`)
- Special characters in input
- Unicode and encoding edge cases
