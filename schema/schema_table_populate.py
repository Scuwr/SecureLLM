import sys
sys.path.append('..')

import sqlite3
import env
import numpy as np

def create(schema_id, filename):
    if schema_id == 1:
        create_1(filename)
    if schema_id == 2:
        create_2(filename)
    if schema_id == 3:
        create_3(filename)
    if schema_id == 4:
        raise NotImplementedError
        create_4(filename)
    if schema_id == 5:
        create_5(filename)
    
def create_1(filename):
    # schema: teachers and courses (2 tables)
    # -teachers table
    # --teacher_id (PK)
    # --name
    # --age (number)

    # -class history table (historical record of classes taught)
    # --class_id (PK)
    # --teacher_id (FK)
    # --level (grade level) (12 / 11 / 10 / ... / 1)
    # --year (year taught) (2019, 2020, etc...)
    # --grade (class average grade) (90 / 77  / 99 / ...)
    # --subject (science / math / english / etc...)

    con = sqlite3.connect(filename)
    cur = con.cursor()

    cur.execute("CREATE TABLE instructors(teacher_id DECIMAL PRIMARY KEY, name TEXT, teacher_age DECIMAL)")
    cur.execute("CREATE TABLE classes("
                            "class_id DECIMAL PRIMARY KEY, "
                            "teacher_id DECIMAL, "
                            "level DECIMAL, "
                            "year DECIMAL, "  # year taught
                            "grade DECIMAL, "  # class average grade
                            "class_subject TEXT "
                            ")")


    first_names = ['John', 'Jane', 'Adam', 'Chris', 'Sally', 'Mike', 'Jill', 'Bob', 'Mary', 'Joe', 'Sue', 'Bill', 'Jen', 'Tom', 'Amy', 'Sam', 'Kim', 'Tim', 'Ann', 'Ron']
    # last_names = ['Smith', 'Johnson', 'Williams', 'Jones', 'Brown', 'Davis', 'Miller', 'Wilson', 'Moore', 'Taylor', 'Anderson', 'Thomas', 'Jackson', 'White', 'Harris', 'Martin', 'Thompson', 'Garcia', 'Martinez', 'Robinson']
    last_names = ['Smith']
    subjects = ['math', 'biology', 'chemistry', 'physics', 'economics', 'history', 'politics', 'philosophy', 'psychology', 'sociology', 'art', 'music', 'english', 'literature', 'poetry']

    cur_teacher_id = 100
    cur_class_id = 10000
    num_teachers = 1000
    num_classes_taught = lambda: max(0, int(np.random.normal(20, 5)))
    get_age = lambda: np.random.randint(20, 70)
    years_taught = lambda: range(*sorted(np.random.choice(range(1980, 2024), 2, replace=False)))
    subjects_taught = lambda: np.random.choice(subjects, max(1, int(np.random.normal(5, 2))), replace=False)
    # grades_taught = lambda: range(*sorted(np.random.choice(range(1, 13), 2, replace=False)))
    grade_dist = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    grades_taught = lambda: np.array([grade_dist[i] for i in np.random.choice(range(4), np.random.randint(4)+1, replace=False)]).flatten()
    teacher_grade_mu = lambda: np.random.normal(70, 5)
    teacher_grade_sigma = lambda: max(1, np.random.normal(8, 4))

    for _ in range(num_teachers):
        cur_teacher_id += 1
        name = f'{np.random.choice(first_names)} {np.random.choice(last_names)}'
        age = get_age()
        cur.execute("INSERT INTO instructors VALUES (?, ?, ?)", (cur_teacher_id, name, age))
        cur_teacher_subjects = subjects_taught()
        cur_teacher_grade_levels = grades_taught()
        cur_teacher_years_taught = years_taught()
        cur_teacher_grade_mu = teacher_grade_mu()
        cur_teacher_grade_sigma = teacher_grade_sigma()
        for _ in range(num_classes_taught()):
            cur_class_id += 1
            subject = np.random.choice(cur_teacher_subjects)
            # must cast from numpy.int64 to int 
            grade_level = int(np.random.choice(cur_teacher_grade_levels))
            year_taught = int(np.random.choice(cur_teacher_years_taught))
            class_avg_grade = round(np.random.normal(cur_teacher_grade_mu, cur_teacher_grade_sigma), 1)
            class_avg_grade = max(0, min(100, class_avg_grade))
            cur.execute("INSERT INTO classes VALUES (?, ?, ?, ?, ?, ?)", (cur_class_id, cur_teacher_id, grade_level, year_taught, class_avg_grade, subject))
    con.commit()
    con.close()


def create_2(filename):
    # schema: appliance (3 tables)
    # -appliance table
    # --appliance_id (PK)
    # --manufacturer “LG, GE, Sony, Samsung, Panasonic”
    # --type “Refrigerator, Dishwasher, Oven, Microwave, Blender, …)
    # --appliance_rating (number,0,10)

    # -store table
    # --store_id (PK)
    # --state (MA, IN, …)
    # --rating (number,0,10)
    # --name (name of store owner, to natural join with schema 1) (string)

    # -inventory table
    # --inventory_id (PK)
    # --appliance_id (FK)
    # --store_id (FK)
    # --width (number)
    # --height (number)
    # --value (number)
    # --available (number,0,1)

    con = sqlite3.connect(filename)
    cur = con.cursor()

    cur.execute("CREATE TABLE appliance(appliance_id DECIMAL PRIMARY KEY, company TEXT, type TEXT, appliance_rating DECIMAL)")
    cur.execute("CREATE TABLE store(store_id DECIMAL PRIMARY KEY, location TEXT, star_rating DECIMAL, name STRING)")
    cur.execute("CREATE TABLE inventory("
                            "inventory_id DECIMAL PRIMARY KEY, "
                            "appliance_id DECIMAL, "
                            "store_id DECIMAL, "
                            "width DECIMAL, "
                            "height DECIMAL, "
                            "value DECIMAL, "
                            "available DECIMAL)")

    first_names = ['John', 'Jane', 'Adam', 'Chris', 'Sally', 'Mike', 'Jill', 'Bob', 'Mary', 'Joe', 'Sue', 'Bill', 'Jen', 'Tom', 'Amy', 'Sam', 'Kim', 'Tim', 'Ann', 'Ron']
    # last_names = ['Smith', 'Johnson', 'Williams', 'Jones', 'Brown', 'Davis', 'Miller', 'Wilson', 'Moore', 'Taylor', 'Anderson', 'Thomas', 'Jackson', 'White', 'Harris', 'Martin', 'Thompson', 'Garcia', 'Martinez', 'Robinson']
    last_names = ['Smith']

    manufacturers = ['LG', 'GE', 'Sony', 'Samsung', 'Panasonic']
    types = ['Refrigerator', 'Dishwasher', 'Oven', 'Microwave', 'Blender']
    states = ['MA', 'IN', 'CA', 'TX', 'NY', 'FL']
    appliance_rating = lambda: round(np.random.normal(5, 2), 1)
    store_rating = lambda: round(np.random.normal(5, 2), 1)
    store_width = lambda: round(np.random.normal(20, 5), 1)
    store_height = lambda: round(np.random.normal(20, 5), 1)
    store_value = lambda: round(np.random.normal(1000, 200), 2)
    store_available = lambda: np.random.randint(0, 2)

    cur_appliance_id = 100
    cur_store_id = 10000
    cur_inventory_id = 100000
    num_appliances = 500
    num_stores = 500
    num_inventories = lambda: max(1, int(np.random.normal(40, 10)))
    for _ in range(num_stores):
        cur_store_id += 1
        state = np.random.choice(states)
        rating = store_rating()
        name = f'{np.random.choice(first_names)} {np.random.choice(last_names)}'

        cur.execute("INSERT INTO store VALUES (?, ?, ?, ?)", (cur_store_id, state, rating, name))
    for _ in range(num_appliances):
        cur_appliance_id += 1
        manufacturer = np.random.choice(manufacturers)
        type_ = np.random.choice(types)
        rating = appliance_rating()
        cur.execute("INSERT INTO appliance VALUES (?, ?, ?, ?)", (cur_appliance_id, manufacturer, type_, rating))
        for _ in range(num_inventories()):
            cur_inventory_id += 1
            appliance_id = np.random.randint(cur_appliance_id)
            store_id = np.random.randint(cur_store_id)
            width = store_width()
            height = store_height()
            value = store_value()
            available = store_available()
            cur.execute("INSERT INTO inventory VALUES (?, ?, ?, ?, ?, ?, ?)", (cur_inventory_id, appliance_id, store_id, width, height, value, available))
    con.commit()
    con.close()

def create_5(filename):
    # schema: publishing houses to books
    # -publishing houses table
    # --publishier_id  (PK)
    # --publishier_name 
    # --state "MA, IN, …"
    # --size (annual rate of books published)
    # --year established 
    # -books table
    # --book_id (PK)
    # --publishing house id
    # --title (FK)
    # --author
    # --year published
    # --genre (fiction, mystery, young adult, classics, …)
    con = sqlite3.connect(filename)
    cur = con.cursor()
    #cur.execute("DROP TABLE IF EXISTS publisher")
    cur.execute("CREATE TABLE publishers(publisher_id  DECIMAL PRIMARY KEY, name TEXT, state TEXT, size DECIMAL, year DECIMAL)")
    #cur.execute("DROP TABLE IF EXISTS book")
    cur.execute("CREATE TABLE books("
                            "book_id  DECIMAL PRIMARY KEY, "
                            "publisher_id DECIMAL, "
                            "title TEXT, "
                            "author TEXT, "
                            "year DECIMAL, "
                            "genre TEXT "
                            ")")
    publisher_names = ['HarperCollins', 'Simon & Schuster', 'Macmillan', 'Penguin', 'Hachette',
                   'Random House', 'Scholastic', 'Wiley', 'Pearson', 'Oxford University Press',
                   'Cengage', 'Bloomsbury', 'Elsevier', 'Harlequin', 'Little, Brown and Company',
                   'Cambridge University Press', 'Pan Macmillan', 'Houghton Mifflin Harcourt', 
                   'Springer', 'Hodder & Stoughton', 'Faber & Faber', 'Walker Books', 'Vintage Books']
    
    publisher_state = ['MA', 'IN', 'CA', 'TX', 'NY', 'FL']
    book_genres = ['Mystery', 'Science Fiction', 'Fantasy', 'Romance', 'Thriller', 'Horror', 'Biography', 'Historical Fiction', 'Non-Fiction',
               'Drama', 'Comedy', 'Action', 'Adventure', 'Poetry', 'Science', 'Self-Help', 'Cookbook', 'Travel', 'Memoir',
               'Graphic Novel', 'Young Adult', 'Children\'s', 'Satire', 'Western', 'Dystopian', 'Manga', 'Crime', 'Espionage']

    book_titles = ['Animal Farm', 'Life of Pi', 'The Bell Jar', 'Othello', '1984', 'To Kill a Mockingbird', 'The Great Gatsby',
               'Pride and Prejudice', 'The Catcher in the Rye', 'War and Peace', 'The Hobbit', 'The Da Vinci Code', 'The Shining',
               'The Alchemist', 'The Hunger Games', 'Harry Potter and the Sorcerer\'s Stone', 'The Lord of the Rings',
               'Brave New World', 'The Chronicles of Narnia', 'The Odyssey', 'The Divine Comedy', 'Dracula', 'Frankenstein']

    book_authors = ['George Orwell', 'Yann Martel', 'Sylvia Plath', 'William Shakespeare', 'Jane Austen', 'J.K. Rowling', 'Leo Tolstoy',
                'Dan Brown', 'Stephen King', 'Paulo Coelho', 'J.R.R. Tolkien', 'Homer', 'Dante Alighieri', 'Bram Stoker', 'Mary Shelley',
                'Agatha Christie', 'Arthur Conan Doyle', 'Jules Verne', 'H.G. Wells', 'Charles Dickens', 'F. Scott Fitzgerald', 'Mark Twain',
                'William Faulkner', 'Emily Brontë']
    
    cur_p_id = 100
    cur_book_id = 1000
    num_publishers = 5
    num_books_published = lambda: int(round(np.random.normal(5000, 500), 0))
    get_year = lambda: np.random.randint(1584, 2023)
    for _ in range(num_publishers):
        cur_p_id += 1
        cur_p_name = f'{np.random.choice(publisher_names)}'
        cur_p_year = get_year()
        cur_p_state = np.random.choice(publisher_state)
        cur_p_size = num_books_published()
        # publisher_id  DECIMAL PRIMARY KEY, name TEXT, state TEXT, size DECIMAL, year DECIMAL
        cur.execute("INSERT INTO publishers VALUES (?, ?, ?, ?, ?)", 
                    (cur_p_id, cur_p_name, cur_p_state, cur_p_size, cur_p_year))
        for _ in range(num_books_published()):
            cur_book_id += 1
            book_title = np.random.choice(book_titles)
            book_author = np.random.choice(book_authors)
            year_published = int(np.random.randint(cur_p_year,2023))
            book_genre = np.random.choice(book_genres)
            cur.execute("INSERT INTO books VALUES (?, ?, ?, ?, ?, ?)", (cur_book_id, cur_p_id,
                                                                       book_title, book_author, year_published, book_genre))
    con.commit()
    con.close()

def create_3(filename):
    '''
    schema: Colleges to Students

    College Table: 
    college_id (PK)
    state (AK, ..., WY)
    type (public, private)
    size 

    Student Table:
    student_id (PK)
    name (name of student, to natural join with schema 1) (string)
    college_id (FK)
    major
    dorm -> random? 
    meal_plan (1,0)
    graduation_year
    GPA
    Course_work
    '''
    con = sqlite3.connect(filename)
    cur = con.cursor()

    cur.execute("CREATE TABLE universities(school_id DECIMAL PRIMARY KEY, territory TEXT, funding TEXT, capacity DECIMAL)")
    cur.execute("CREATE TABLE pupils(pupil_id DECIMAL PRIMARY KEY, name TEXT, school_id DECIMAL, study TEXT, housing TEXT, cafeteria INT, end_date DECIMAL, performance DECIMAL, classes TEXT)")

    # cur.execute("CREATE TABLE colleges(college_id DECIMAL PRIMARY KEY, state TEXT, type TEXT, size DECIMAL)")
    # cur.execute("CREATE TABLE students(student_id DECIMAL PRIMARY KEY,name TEXT, college_id DECIMAL, major TEXT, dorm TEXT, meal_plan INT, graduation_year DECIMAL, GPA DECIMAL, course_work TEXT)")

    # Preparation for college table
    school_size = lambda: 1000+ np.random.beta(1.5,1.7)*20000 # Use beta distribution which more accurately represents college population
    states = [ 'AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA',
           'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME',
           'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM',
           'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX',
           'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY'] # Use random.choice

    # Preparation for student table
    first_names = ['John', 'Jane', 'Adam', 'Chris', 'Sally', 'Mike', 'Jill', 'Bob', 'Mary', 'Joe', 'Sue', 'Bill', 'Jen', 'Tom', 'Amy', 'Sam', 'Kim', 'Tim', 'Ann', 'Ron']
    # last_names = ['Smith', 'Johnson', 'Williams', 'Jones', 'Brown', 'Davis', 'Miller', 'Wilson', 'Moore', 'Taylor', 'Anderson', 'Thomas', 'Jackson', 'White', 'Harris', 'Martin', 'Thompson', 'Garcia', 'Martinez', 'Robinson']
    last_names = ['Smith']

    dorms = ["Anderson Hall", "Baker House", "Carson Hall", "Davidson Hall", "Edwards House", "Franklin Hall", "Grant House", "Harris Hall", 
        "Irving House", "Johnson Hall", "Keller House", "Lincoln Hall", "Morgan House", "Nelson Hall", "O'Connor House", "Parker Hall", 
        "Quinn House", "Russell Hall", "Sullivan House", "Thompson Hall", "Underwood House", "Vincent Hall", "Wallace House", "Young Hall", 
        "Zimmerman House", "Adams Hall", "Bradley House", "Campbell Hall", "Davis House", "Elliott Hall", "Fisher House", "Green Hall", 
        "Harper House", "Ingram Hall", "King House", "Lawson Hall", "Mitchell House", "Norton Hall", "Owens House", "Patterson Hall"] # 50 randomly generated dorms
    graduation_years = [2024, 2025, 2026, 2027]
    get_gpa = lambda: min(4.0, max(1.0, round(np.random.normal(3, 1), 1))) # 4.0 GPA scale
    majors = ['Math', 'Biology', 'Chemistry', 'Physics', 'Economics', 'History', 'Political Science', 
                'Philosophy', 'Psychology', 'Sociology', 'Art', 'Music', 'English', 'Classics',
                'Computer Science', 'Mechanical Engineering']
    courses = [
    "Differential Equations", "Introduction to Programming", "Dynamics", "French Language and Culture", "German Literature", "English Composition", "Organic Chemistry", "Principles of Economics", "World History", 
    "Social Psychology", "Introduction to Philosophy", "Business Ethics", "Computer Graphics", "Data Structures and Algorithms", "Linear Algebra", "General Physics", "Cell Biology", "American Literature", "Spanish for Beginners", "Environmental Science", "Art History", "Digital Marketing", "Microbiology", 
    "Statistics for Engineers", "Calculus II", "Modern Political Thought", "Introduction to Sociology", "Cognitive Neuroscience", "Creative Writing", "Theories of Personality", "Quantum Mechanics", "Fundamentals of Nursing", "Graphic Design", "Public Speaking", "Human Anatomy", "Astronomy", 
    "Music Theory", "Foundations of Education", "Geology", "Macroeconomics", "Biochemistry", "Comparative Politics", "Physical Education", "Ceramics", "Software Engineering", "Philosophy of Science", "Game Theory", "Photography", "International Relations", "Zoology", "Entrepreneurship", "Ethics in Technology", 
    "Forensic Science", "Modern Architecture", "Abnormal Psychology", "Human Geography", "Robotics", "Global Health", "Theatre and Drama", "Documentary Filmmaking", "Medieval History", "Genetics", "Cultural Anthropology", "African American Studies", "Human Resource Management", "Urban Planning", "Introduction to Astronomy", 
    "Nutritional Science", "Renaissance Art", "Cybersecurity Fundamentals", "Ancient Philosophy", "Molecular Biology", "Environmental Law", "Victorian Literature", "Digital Media Production", "Asian Studies", "Classical Mechanics", "Contemporary Poetry", "Financial Accounting", "Marine Biology", "Logic and Critical Thinking", 
    "Introduction to Theology", "Fashion Design", "Russian Language and Literature", "Neurobiology", "Journalism", "Supply Chain Management", "Child Development", "Medieval Literature", "Organizational Behavior", "Introduction to Meteorology", "Political Economy", "Introduction to Film Studies", "Biostatistics", 
    "Latin American Studies", "Web Development", "Physical Chemistry", "Public Policy Analysis", "Modern Dance", "Advanced Calculus", "Women's Studies", "Materials Science", "Veterinary Medicine", "Introduction to Robotics", "Health Psychology", "Agricultural Science", "Real Estate Finance", "Mythology and Folklore"
] # 100 random classes
    num_courses = {2024: 16, 2025: 12, 2026: 8, 2027: 4}
    cur_college_id = 1
    num_colleges = 500
    cur_student_id = 1000
    num_students = 500

    for _ in range(num_colleges):
        state = np.random.choice(states)
        type = 'private' if np.random.random() > .5 else 'public'
        size = round(school_size())

        cur.execute('INSERT INTO universities VALUES (?, ?, ?, ?)', (cur_college_id, state, type, size))
        cur_college_id +=1
    
    for _ in range(num_students):
        name = np.random.choice(first_names) + ' ' + np.random.choice(last_names)
        college_id = np.random.randint(1, num_colleges+1)
        major = np.random.choice(majors)
        dorm = np.random.choice(dorms)
        meal_plan = 1 if np.random.random() > .5 else 0
        graduation_year = int(np.random.choice(graduation_years))
        gpa = get_gpa()
        course_work = ', '.join(np.random.choice(courses, num_courses[graduation_year], replace = False))

        cur.execute("INSERT INTO pupils VALUES (?,?,?,?,?,?,?,?,?)", (cur_student_id, name, college_id, major, dorm, meal_plan, graduation_year, gpa, course_work))
        cur_student_id +=1

    con.commit()
    con.close()

if __name__ == '__main__':
    import os
    print(os.getcwd())
    # os.makedirs('./schema_data', exist_ok=True)
    # for i in range(5):
    #     create_1(f'./schema_data/schema_1_{i}.db')
    # for i in range(5):
    #     create_2(f'./schema_data/schema_2_{i}.db')
    # for i in range(5):
    #     create_3(f'./training_data/databases/schema_3_{i}.db')
    # for i in range(5):
    #     create_5(f'./schema_data/schema_5_{i}.db')

    # UNION
    # for i in range(5):
    #     fn = env.dataset_paths.schema_databases / f'schema_union12_{i}.db'
    #     print(1)
    #     create_1(fn)
    #     print(2)
    #     create_2(fn)

    for i in range(5):
        fn = env.dataset_paths.schema_databases / f'schema_union13_{i}.db'
        print(1)
        create_1(fn)
        print(3)
        create_3(fn)

    for i in range(5):
        fn = env.dataset_paths.schema_databases / f'schema_union23_{i}.db'
        print(2)
        create_2(fn)
        print(3)
        create_3(fn)

    # for i in range(5):
    #     fn = env.dataset_paths.schema_databases / f'schema_union123_{i}.db'
    #     print(1)
    #     create_1(fn)
    #     print(2)
    #     create_2(fn)
    #     print(3)
    #     create_3(fn)


    # con = sqlite3.connect('./schema_data/schema_2_0.db')
    # cur = con.cursor()
    # print(cur.execute("SELECT * FROM store LIMIT 10").fetchall())
