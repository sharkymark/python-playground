# SQL injection example

This is a simple SQL Injection example where the user input is not validated

## The app

Start the app with `python3 customer.py` to create a SQLite3 database. After you enter a filter criteria, the app prints out the SQL statement. You can see how it's exploited using the attempts below.

## Force SQL injection

At the last name input prompt, enter:

1. `%%`
1. `'OR 1=1 --`
1. `' UNION SELECT * FROM customers --`
1. `' OR 1=1; DELETE FROM customers; --`

## Resources

[Python Mac versions](https://www.python.org/downloads/macos/)

[dev container spec](https://containers.dev/implementors/json_reference/)

## License

This project is licensed under the [MIT License](LICENSE)

## Contributing

### Disclaimer: Unmaintained and Untested Code

Please note that this program is not actively maintained or tested. While it may work as intended, it's possible that it will break or behave unexpectedly due to changes in dependencies, environments, or other factors.

Use this program at your own risk, and be aware that:
1. Bugs may not be fixed
1. Compatibility issues may arise
1. Security vulnerabilities may exist

If you encounter any issues or have concerns, feel free to open an issue or submit a pull request.