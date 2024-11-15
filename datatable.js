const DataTableManager = (function () {
    const instances = {};

    function initializeDataTable({ rows, tableId, tableHeadId, filterRowId, columnTitleMap, callback }) {
        if (!Array.isArray(rows)) {
            console.error("Error: Rows must be an array.");
            return;
        }

        // Determine columns dynamically or fallback to a placeholder for empty rows
        const columns = rows.length > 0
            ? Object.keys(rows[0]).map(key => ({
                data: key,
                title: columnTitleMap[key] || key,
                defaultContent: '' // Avoid undefined errors
            }))
            : [{ data: null, title: 'No Data', defaultContent: 'No records to display' }];

        console.log("Rows: ", rows);
        console.log("Columns: ", columns);

        // Check if the table already exists
        if (instances[tableId]) {
            const dataTable = instances[tableId];

            // Clear and re-add rows
            dataTable.clear();
            dataTable.rows.add(rows);
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
            data: rows,
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
