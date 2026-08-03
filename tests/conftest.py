def pytest_configure(config):
    config.addinivalue_line("markers", "integration: requires the Compose platform")
