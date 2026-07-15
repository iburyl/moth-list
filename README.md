# moths-list

A stub Django project.

## Requirements

- Python 3.10+
- pip

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Running

Apply migrations and start the development server:

```bash
python manage.py migrate
python manage.py runserver
```

The site will be available at http://127.0.0.1:8000/ and the admin at
http://127.0.0.1:8000/admin/.

Create an admin user with:

```bash
python manage.py createsuperuser
```
