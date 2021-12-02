def get_logger(*, logger_name=None, level="DEBUG"):
    import logging

    FORMAT = "%(asctime)s [%(levelname)s] %(filename)s:%(funcName)s-L%(lineno)d: %(message)s"
    logging.basicConfig(format=FORMAT)
    log = logging.getLogger(logger_name)
    log.setLevel(level)
    return log
