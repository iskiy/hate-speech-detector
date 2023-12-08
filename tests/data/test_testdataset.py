def test_dataset(df):
    """Test dataset quality and integrity."""
    # missing values
    df.expect_column_values_to_not_be_null(column="id")
    df.expect_column_values_to_not_be_null(column="comment_text")

    # unique values
    df.expect_column_values_to_be_unique(column="id")
    df.expect_column_values_to_be_unique(column="comment_text")

    # type adherence
    df.expect_column_values_to_be_of_type(column="comment_text", type_="str")

    # Expectation suite
    expectation_suite = df.get_expectation_suite(discard_failed_expectations=False)
    results = df.validate(
        expectation_suite=expectation_suite, only_return_failures=True
    ).to_json_dict()
    assert results["success"]
