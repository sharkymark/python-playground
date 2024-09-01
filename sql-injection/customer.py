import sqlite3
import os

# Create a new database if it doesn't exist
if not os.path.exists('customer.db'):
    conn = sqlite3.connect('customer.db')
    c = conn.cursor()

    # Create table
    c.execute('''CREATE TABLE customers
                 (first_name text, last_name text, ssn text, birth_date text, email text, phone text, credit_card_num text)''')

    # Insert 10 rows of sample data
    sample_data = [
        ('Brad', 'Pitt', '123-45-6789', '1960-01-01', 'brad@example.com', '123-456-7890', '1234-5678-9012-3456'),
        ('Katie', 'Perry', '987-65-4321', '1991-02-02', 'kp@example.com', '987-654-3210', '9876-5432-1098-7654'),
        ('Tom', 'Hanks', '456-78-9123', '1956-03-03', 'tomh@gmail.com', '456-789-1230', '4567-8912-3456-7891'),
        ('Angelina', 'Jolie', '789-12-3456', '1975-04-04', 'ajoie@aol.com', '789-123-4560', '7891-2345-6789-1234'),
        ('Mark','Milligan','654-32-1098','1980-05-05','mtm@gmai.com','654-321-0980','6543-2109-8765-4321')
    ]
    c.executemany('INSERT INTO customers VALUES (?, ?, ?, ?, ?, ?, ?)', sample_data)
    conn.commit()
    conn.close()

# Connect to the database
conn = sqlite3.connect('customer.db')
c = conn.cursor()

# Prompt the user to enter their last name
last_name = input('Enter your last name: ')

# SQL query with user input
#query = f"SELECT * FROM customers"
query = f"SELECT * FROM customers WHERE last_name LIKE '{last_name}'"

print(f"\nExecuting query: {query}\n")
# Execute the query
c.execute(query)

# Fetch and print all rows
print("\nCustomer info:\n")
rows = c.fetchall()
for row in rows:
    print(row)

# Close the connection
conn.close()