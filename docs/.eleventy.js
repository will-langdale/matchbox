const govukEleventyPlugin = require("@x-govuk/govuk-eleventy-plugin");
const fs = require("fs");

module.exports = function (eleventyConfig) {
  eleventyConfig.addPlugin(govukEleventyPlugin, {
    showBreadcrumbs: false,
    icons: {
      shortcut: "/src/assets/dbt-favicon.png",
      scssSettingsPath: "/src/assets/sass/_settings.scss",
    },
    navigation: {
      items: [
        {
          text: "Client",
          href: "/client",
        },
        {
          text: "Server",
          href: "/server",
        },
        {
          text: "Contributing",
          href: "/contributing",
        },
      ],
    },
    header: {
      logotype: {
        html: fs.readFileSync("./src/assets/dbt-logo.svg", {
          encoding: "utf8",
        }),
      },
      productName: "Matchbox",
    },
    footer: {
      meta: {
        items: [
          {
            href: "https://github.com/uktrade/pg-bulk-ingest",
            text: "GitHub repository for pg-bulk-ingest",
          },
          {
            href: "https://www.gov.uk/government/organisations/department-for-business-and-trade",
            text: "Created by the Department for Business and Trade (DBT)",
          },
        ],
      },
    },
  });

  eleventyConfig.addPassthroughCopy("./src/assets");
  // eleventyConfig.addPassthroughCopy("./docs/CNAME");

  return {
    dataTemplateEngine: "njk",
    htmlTemplateEngine: "njk",
    markdownTemplateEngine: "njk",
    dir: {
      input: "src",
      output: "_site",
      layouts: "../node_modules/@x-govuk/govuk-eleventy-plugin/layouts",
    },
  };
};
