def test_output(df):
    """Test output."""
    # schema adherence
    column_list = ["id", "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    df.expect_table_columns_to_match_ordered_list(column_list=column_list)

    # expected labels
    labels = [0, 1]
    df.expect_column_values_to_be_in_set(column="toxic", value_set=labels)
    df.expect_column_values_to_be_in_set(column="severe_toxic", value_set=labels)
    df.expect_column_values_to_be_in_set(column="obscene", value_set=labels)
    df.expect_column_values_to_be_in_set(column="threat", value_set=labels)
    df.expect_column_values_to_be_in_set(column="insult", value_set=labels)
    df.expect_column_values_to_be_in_set(column="identity_hate", value_set=labels)

    # missing values
    df.expect_column_values_to_not_be_null(column="id")
    df.expect_column_values_to_not_be_null(column="toxic")
    df.expect_column_values_to_not_be_null(column="severe_toxic")
    df.expect_column_values_to_not_be_null(column="obscene")
    df.expect_column_values_to_not_be_null(column="threat")
    df.expect_column_values_to_not_be_null(column="insult")
    df.expect_column_values_to_not_be_null(column="identity_hate")

    # unique values
    df.expect_column_values_to_be_unique(column="id")

    # Expectation suite
    expectation_suite = df.get_expectation_suite(discard_failed_expectations=False)
    results = df.validate(
        expectation_suite=expectation_suite, only_return_failures=True
    ).to_json_dict()
    assert results["success"]