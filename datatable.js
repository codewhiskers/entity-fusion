const DataTableManager = (function () {
    const instances = {};

    function initializeDataTable({ rows, tableId, tableHeadId, filterRowId, columnTitleMap, callback }) {
        if (!Array.isArray(rows)) {
            console.error("Error: Rows must be an array.");
            return;
        }

        // Filter the rows to include only the specified columns (plus `id` for interactivity)
        const filteredRows = rows.map(row => {
            const filteredRow = {};
            Object.keys(columnTitleMap).forEach(key => {
                filteredRow[key] = row[key]; // Include columns specified in columnTitleMap
            });
            filteredRow["id"] = row["id"]; // Always include the `id` column for interactivity
            return filteredRow;
        });

        // Generate column definitions based on columnTitleMap
        const columns = Object.keys(columnTitleMap).map(key => ({
            data: key,
            title: columnTitleMap[key] || key,
            defaultContent: '' // Avoid undefined errors
        }));

        // Always include the `id` column as a hidden column
        columns.push({
            data: "id",
            title: "Node ID (Hidden)",
            visible: false, // Hide the `id` column in the table
            defaultContent: ''
        });

        console.log("Filtered Rows: ", filteredRows);
        console.log("Columns: ", columns);

        // Check if the table already exists
        if (instances[tableId]) {
            const dataTable = instances[tableId];

            // Clear and re-add rows
            dataTable.clear();
            dataTable.rows.add(filteredRows);
            dataTable.draw();
            return; // Skip the rest of the initialization process
        }

        // Set up the table header and filter row dynamically
        const tableHead = document.getElementById(tableHeadId);
        const filterRow = document.getElementById(filterRowId);

        tableHead.innerHTML = '';
        filterRow.innerHTML = '';

        columns.forEach((col, index) => {
            const displayTitle = col.title;

            // Create header for each column
            const th = document.createElement('th');
            th.textContent = displayTitle;
            tableHead.appendChild(th);

            // Create header filter input dynamically
            const filterTh = document.createElement('th');
            const filterInput = document.createElement('input');
            filterInput.type = 'text';
            filterInput.placeholder = `Search ${displayTitle}`;
            filterInput.style.width = '100%';
            filterInput.setAttribute('data-index', index);
            filterTh.appendChild(filterInput);
            filterRow.appendChild(filterTh);

            // Attach event listener for filter input
            filterInput.addEventListener('keyup', function () {
                const colIndex = filterInput.getAttribute('data-index');
                const searchTerm = filterInput.value;

                instances[tableId]
                    .column(colIndex)
                    .search(searchTerm)
                    .draw();
            });
        });

        // Initialize DataTable
        const dataTable = $(`#${tableId}`).DataTable({
            data: filteredRows,
            columns: columns,
            paging: true,
            searching: true,
            info: false,
            scrollCollapse: false,
            scrollX: true,
            scrollY: '100%',
            responsive: false,
            fixedColumns: true,
            orderCellsTop: true,
            fixedHeader: true,
            pageLength: 25,
            select: true
        });

        instances[tableId] = dataTable;

        if (callback) {
            callback(dataTable);
        }

        $(window).trigger('resize');
    }

    function getInstance(tableId) {
        return instances[tableId];
    }

    return {
        initializeDataTable,
        getInstance
    };
})();
