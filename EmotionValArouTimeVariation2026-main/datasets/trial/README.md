## Please register for the shared task on [codabench competition page](https://www.codabench.org/competitions/9963/) to access the datasets.

### Please refer to our [website](https://semeval2026task2.github.io/SemEval-2026-Task2/) and [codabench competition page](https://www.codabench.org/competitions/9963/) for more details on the shared task and how to participate.

# Data Format

Trial data files are shared in both subtasks.


<table style="border-collapse:collapse;width:100%;border:1px solid #ccc;font-size:14px;">
  <thead>
    <tr style="background:#f7f7f7;">
      <th style="padding:6px 8px;border:1px solid #ccc;"><strong>user_id</strong></th>
      <th style="padding:6px 8px;border:1px solid #ccc;"><strong>text_id</strong></th>
      <th style="padding:6px 8px;border:1px solid #ccc;"><strong>text</strong></th>
      <th style="padding:6px 8px;border:1px solid #ccc;"><strong>timestamp</strong></th>
      <th style="padding:6px 8px;border:1px solid #ccc;"><strong>collection_phase</strong></th>
			<th style="padding:6px 8px;border:1px solid #ccc;"><strong>is_words</strong></th>
      <th style="padding:6px 8px;border:1px solid #ccc;"><strong>valence</strong></th>
      <th style="padding:6px 8px;border:1px solid #ccc;"><strong>arousal</strong></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding:6px 8px;border:1px solid #ccc;"><em>example_user_id</em></td>
      <td style="padding:6px 8px;border:1px solid #ccc;"><em>example_text_id</em></td>
      <td style="padding:6px 8px;border:1px solid #ccc;">example text</td>
      <td style="padding:6px 8px;border:1px solid #ccc;">example timestamp</td>
      <td style="padding:6px 8px;border:1px solid #ccc;">[1, 2, 3, 4, 5, 6]</td>
			<td style="padding:6px 8px;border:1px solid #ccc;">[True, False]</td>
      <td style="padding:6px 8px;border:1px solid #ccc;">[0, 1, 2, 3, 4]</td>
      <td style="padding:6px 8px;border:1px solid #ccc;">[0, 1, 2]</td>
    </tr>
  </tbody>
</table>

Note 1: The data was collected in 7 different phases over multiple years (2021–2024). We provide the <code>collection_phase</code> as additional information in case it is helpful.

Note 2: The participants in the dataset collection study could describe "how they are feeling" by writing an essay or writing 1 to 5 feeling words (e.g., happy, calm, sad, etc.). We provide a boolean column <code>is_words</code> to distinguish the two forms.
