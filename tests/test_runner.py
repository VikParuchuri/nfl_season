"""
Runs tests
"""

from percept.utils.registry import registry
from percept.conf.base import settings
import logging

log = logging.getLogger(__name__)

def run_all_tests():
    """
    Look through the registry, and run tests for any class that has a tester and test_cases
    """
    for item in registry:
        item_cls = item.cls
        if hasattr(item_cls, 'tester') and hasattr(item_cls, 'test_cases'):
            tester = item_cls.tester()
            yield tester.run, item_cls, item_cls.test_cases

